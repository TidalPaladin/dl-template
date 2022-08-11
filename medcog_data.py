#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Sequence, Iterable, Iterator

import albumentations as A
import numpy as np
import torch
from combustion.data import TorchDataset  # type: ignore
from torch import Tensor
from itertools import cycle

from medcog_preprocessing import from_int16
from medcog_preprocessing.data import LookupHandler, ProcessedImage, Target
from medcog_preprocessing.types import MISSING, Embeddings
from torch.utils.data import Subset, Dataset, WeightedRandomSampler

from ..structs import CaseExample, ImageExample, RegionalLabel


nn_malign_embedding: Final[int] = 1
nn_benign_embedding: Final[int] = 2
embeddings: Final[Embeddings] = Embeddings()

Batch = Tuple[Tensor, Dict[str, Tensor], List[str]]


class MedCogDataset(TorchDataset):

    def __init__(self, path: Path, *args, transforms=None, transform=None, **kwargs):
        if not path.parent.is_dir():
            raise NotADirectoryError(path.parent)
        if path.is_dir() and not any(path.rglob("*.pth")):
            raise FileNotFoundError(f"{path} had no .pth files")
        transform = transforms or transform
        super(MedCogDataset, self).__init__(str(path), *args, transform=transform, **kwargs)

    def __getitem__(self, pos: int) -> ImageExample:
        filename = self.files[pos]
        return self.get_transformed_example(filename)

    def get_transformed_example(self, filename: Path) -> ImageExample:
        example = self.load_example(filename)
        example = self.apply_transform(example)

        global_target = example.global_label
        if global_target[1] == 1:
            assert global_target[0] == 1
            if example.annotation.is_malignant:
                assert global_target[1] == 1

        return example

    def apply_transform(self, example: ImageExample) -> ImageExample:
        if self.transform is not None:
            example = self.transform(example)

        if example.regional_label is not MISSING:
            if example.regional_label.coords is MISSING:
                example = example.replace(regional_label=MISSING)
        return example

    @staticmethod
    def load_example(filename: Path) -> ImageExample:
        example = torch.load(filename, map_location="cpu")
        processed_image, (regional_target, global_target) = example
        series = getattr(example, "series", MISSING)
        annotation = getattr(example, "annotation", MISSING)

        if annotation is None:
            annotation = MISSING
        if series is None:
            series = MISSING

        if series:
            name = Path(series.path)
        else:
            name = Path(filename)

        case_id = str(Path(*name.parts[-2:]))

        if global_target[1] == 1:
            assert global_target[0] == 1
            if example.annotation.is_malignant:
                assert global_target[1] == 1

        image = from_int16(processed_image.pixels)
        if global_target is not None:
            global_target = global_target.float()
        else:
            global_target = MISSING

        if regional_target is not None:
            coords = regional_target[..., :-2]
            classes = regional_target[..., -2:]
            regional_label = RegionalLabel(coords, classes)
        else:
            regional_label = MISSING

        example = ImageExample(
            image,
            global_target,
            regional_label,
            source=name,
            case_id=case_id,
            annotation=annotation,
            series=series
        )


        assert example.case_id is not MISSING
        assert example.case_id_hashes is not MISSING
        return example


class MedCogDatasetSplit:
    """Manage a dataset which has been split into malignant, benign, and unknown subfolders"""

    def __init__(self, path: Path, transforms: Optional[Any] = None) -> None:
        def create_dataset(subfolder: str) -> MedCogDataset:
            return MedCogDataset(Path(path, "torch", subfolder), transform=transforms)

        self.maligns = create_dataset("malignant")
        self.benigns = create_dataset("benign")
        self.unknown = create_dataset("unknown")
        self.num_maligns = len(self.maligns)
        self.num_benigns = len(self.benigns)
        self.num_unknowns = len(self.unknown)
        self.num_benign_unknown = self.num_benigns + len(self.unknown)
        self.num_total = self.num_maligns + self.num_benign_unknown
        assert len(self)

    @property
    def files(self) -> Iterator[Path]:
        for f in self.maligns.files:
            yield f
        for f in self.benigns.files:
            yield f
        for f in self.unknown.files:
            yield f


    def __getitem__(self, i: int) -> ImageExample:
        if i < self.num_benigns:
            return self.benigns[i]
        elif i < self.num_benign_unknown:
            return self.unknown[i - self.num_benigns]
        else:
            return self.maligns[i - self.num_benign_unknown]

    def __len__(self):
        return self.num_total


class MedCogDatasetBalanced(MedCogDatasetSplit):
    """Return a balanced split of malignant and benign samples per epoch even though there are likely
    more benign samples available."""

    def __init__(self, path: Path, transforms: Optional[Any] = None, shuffle_benigns: bool = True):
        super(MedCogDatasetBalanced, self).__init__(path=path, transforms=transforms)
        self.path = path
        self.benign_unknown_indices = []
        self.shuffle_benigns = shuffle_benigns

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path}, len={len(self)})"

    def __getitem__(self, i: int) -> ImageExample:
        if i < self.num_maligns:
            return self.maligns[i]
        else:
            return self.get_random_benign_unknown()

    def get_random_benign_unknown(self):
        # Randomly sample from the pool of benigns and unknowns but don't pick the same sample twice until all other
        # samples have been returned at least once
        if self.benign_unknown_indices == []:
            self.benign_unknown_indices = list(range(self.num_benign_unknown))
            if self.shuffle_benigns:
                random.shuffle(self.benign_unknown_indices)
        i = self.benign_unknown_indices.pop()
        return self.benigns[i] if i < self.num_benigns else self.unknown[i - self.num_benigns]

    def __len__(self):
        # For a single epoch we will return every malignant sample once along with a random set of benign samples to
        # match the number of malignant samples
        return self.num_maligns * 2


def transforms(
    p: ProcessedImage, target: Optional[Target] = None, train: bool = False
) -> Tuple[Tensor, Dict[str, Tensor]]:
    image = normalize_image(p.pixels)

    if target is None:
        global_targets = torch.Tensor([-1, -1, -1])
        boxes, labels = torch.Tensor([]).view(0, 4), torch.Tensor([])
    else:
        regional_targets, global_targets = target
        boxes, labels = split_target(regional_targets)

    if train:
        image, boxes, labels = train_transforms(image, boxes, labels)

    target_dict = prepare_target(image, boxes, labels)
    target_dict["abnormalcy"] = torch.Tensor([-1 if target is None else int(boxes.shape[0] != 0)])
    # We could also assume that anything "unknown" is "malignant"
    # Global targets have an order of abnormalcy, malignancy, density
    target_dict["malignancy"] = global_targets[1:2]
    target_dict["density"] = remap_density(int(global_targets[2]))

    return image, target_dict


def prepare_target(image: Tensor, boxes: Tensor, labels: Tensor) -> Dict[str, Tensor]:
    if boxes.shape[0] == 0:
        boxes = torch.Tensor([[-1] * 4])
        labels = torch.Tensor([-1])

    num_rows, num_cols = image.shape[-2:]
    image_scale = torch.Tensor([1])
    image_size = torch.Tensor([num_cols, num_rows])
    boxes = xyxy_to_yxyx(boxes)

    return {
        "bbox": boxes,
        "cls": labels,
        "img_scale": image_scale,
        "img_size": image_size,
    }


def split_target(t: Tensor) -> Tuple[Tensor, Tensor]:
    # A single trace with two types such as asymmetry and distortion will result in two identical bboxes. If those
    # types have been mapped to the same embedding (i.e. 1), then they will result in redundant entries.
    # However, the network seems to train better with these redundant boxes, perhaps because they give more weight
    # to "important" findings with many types (asymmetry, distortion, calcification, etc.)
    # TODO Look into this some more
    # t = torch.unique(t, dim=0)  # Remove redundant bboxes

    bboxes = t[..., :4]
    labels = t[..., 4]
    maligs = t[..., 5]

    mass_bboxes, mass_maligs = get_masses(bboxes, labels, maligs)
    cltr_bboxes, cltr_maligs = get_malig_clusters(bboxes, labels, maligs)

    bboxes = torch.cat((mass_bboxes, cltr_bboxes))
    maligs = torch.cat((mass_maligs, cltr_maligs))

    labels = maligs_to_labels(maligs)

    return bboxes, labels


def maligs_to_labels(maligs: Tensor) -> Tensor:
    if len(maligs) > 0:
        malignant_rois = maligs == embeddings.forward_pathology_embed("Malignant")
        unknown_rois = maligs == embeddings.forward_pathology_embed("Unknown")

        # We want to flag anything that is not definitely benign
        # TODO Is this really a good strategy?
        malignant_rois = torch.logical_or(malignant_rois, unknown_rois)

        labels = torch.Tensor([0] * len(maligs)).type_as(maligs)
        labels[~malignant_rois] = nn_benign_embedding
        labels[malignant_rois] = nn_malign_embedding
        return labels
    else:
        return torch.Tensor([])


def get_masses(bboxes: Tensor, labels: Tensor, maligs: Tensor) -> Tuple[Tensor, Tensor]:
    valid_rois = labels == embeddings.forward_type_embed("Mass")
    bboxes = bboxes[valid_rois]
    maligs = maligs[valid_rois]
    return bboxes, maligs


def get_malig_clusters(bboxes: Tensor, labels: Tensor, maligs: Tensor) -> Tuple[Tensor, Tensor]:
    calc_rois = labels == embeddings.forward_type_embed("Calcification")
    malig_rois = maligs == embeddings.forward_pathology_embed("Malignant")
    valid_rois = torch.logical_and(calc_rois, malig_rois)

    if sum(valid_rois) > 0:
        # TODO Could consider intelligent clustering of malignant calcifications instead of one giant cluster
        bbox = combine_bboxes(bboxes[valid_rois])
        malig = [1]
    else:
        bbox = []
        malig = []

    maligs = torch.Tensor(malig).type_as(maligs)
    bboxes = torch.Tensor(bbox).type_as(bboxes).view(-1, 4)

    return bboxes, maligs


def remap_density(density: int) -> Tensor:
    """Remap densities 1/2 to 0 and 3/4 to 1."""
    density = density if density == -1 else int(density > 2)
    return torch.Tensor([density])


def combine_bboxes(bboxes: Tensor) -> Tensor:
    """Return the smallest possible bbox that contains all input bboxes"""
    x_min = bboxes[:, 0].min()
    y_min = bboxes[:, 1].min()
    x_max = bboxes[:, 2].max()
    y_max = bboxes[:, 3].max()
    bbox = torch.Tensor([x_min, y_min, x_max, y_max]).type_as(bboxes)
    return bbox


def normalize_image(i: Tensor) -> Tensor:
    # TODO It'd probably be better to call from_int16() from medcog_preprocessing
    i = i.float()
    i -= i.min()
    i /= i.max()
    return i


def xyxy_to_yxyx(bboxes: Tensor) -> Tensor:
    bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
    return bboxes


def train_transforms(image: Tensor, boxes: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    image = image.permute(1, 2, 0).numpy()  # CxHxW -> HxWxC
    no_rois = (labels == -1).all() or len(boxes) == 0

    if no_rois:
        sample = albumentations_transform(image=image, bboxes=[], labels=[])
    else:
        # TODO Do we have to convert everything to numpy and then back to tensor?
        # Albumentations doesn't like samples with bbox == [-1, -1, -1, -1]
        valid_samples = labels != -1
        boxes = boxes[valid_samples].numpy()
        labels = labels[valid_samples].numpy()

        # Albumentations will complain if x_min == x_max and y_min == y_max
        rows_match = boxes[:, 0] == boxes[:, 2]
        cols_match = boxes[:, 1] == boxes[:, 3]
        num_rows, num_cols, num_chns = image.shape
        boxes[rows_match] = (boxes[rows_match] + [-1, 0, 1, 0]).clip(0, num_rows - 1)
        boxes[cols_match] = (boxes[cols_match] + [0, -1, 0, 1]).clip(0, num_cols - 1)

        sample = albumentations_transform(image=image, bboxes=boxes, labels=labels)
        augmented_boxes = [torch.Tensor(bbox) for bbox in sample["bboxes"]]
        boxes = torch.Tensor([]).view(0, 4) if augmented_boxes == [] else torch.stack(augmented_boxes)
        labels = torch.Tensor(sample["labels"])

    image = torch.Tensor(sample["image"]).permute(2, 0, 1)  # HxWxC -> CxHxW
    return image, boxes, labels


class InvertColor(A.Lambda):
    """Maximum pixel value becomes 0 and 0 becomes maximum value."""

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super(InvertColor, self).__init__(image=self.invert_color, always_apply=always_apply, p=p)  # type: ignore

    @staticmethod
    def invert_color(image: np.ndarray, **kwargs) -> np.ndarray:
        return 1 - image


class MedCogCaseDataset(MedCogDataset):

    def __init__(self, path: Path, transforms: Optional[Any] = None, **kwargs) -> None:
        if not path.parent.is_dir():
            raise NotADirectoryError(path.parent)
        if path.is_dir() and not any(path.rglob("*.pth")):
            raise FileNotFoundError(f"{path} had no .pth files")
        super().__init__(path, transforms=transforms, **kwargs)
        self.path = path
        self.lookup = self.load_lookup()
        self.cases = self.find_cases()

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, pos: int) -> CaseExample:
        paths = self.cases[pos]
        examples = [self.get_transformed_example(p) for p in paths]
        case_example = CaseExample.from_split(examples)
        return case_example

    def load_lookup(self) -> Optional[LookupHandler]:
        path = Path(self.path, "lookup.pth")
        if path.is_file():
            return torch.load(path)
        return None

    def find_cases(self) -> List[List[Path]]:
        assert self.lookup
        lookup = self.lookup

        result = []
        for study_uid in lookup.study_uid.keys():
            lookup_results = lookup.lookup_study_uid(study_uid)
            paths = [r.preprocessed_path for r in lookup_results]
            result.append(paths)
        return result
