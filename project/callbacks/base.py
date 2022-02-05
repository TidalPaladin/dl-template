#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod, abstractmethod
from functools import wraps
from queue import PriorityQueue
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

from ..metrics import PrioritizedItem, QueueStateCollection
from ..model.base import BaseModel
from ..structs import Example, I, Mode, ModeGroup, O, Prediction, State


ALL_MODES: ModeGroup = ["train", "val", "test"]


def unpack_dict(wrapped):
    r""":class:`LightningModule` is required to return either a loss tensor or a dictionary
    for automatic optimization. This decorator unpacks a returned dictionary by looking for a
    :class:`Prediction` instance under the key `pred`
    """

    @wraps(wrapped)
    def func(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: Dict[str, Any],
        batch: Example,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if isinstance(outputs, dict):
            assert "pred" in outputs.keys(), "returned dictionary should contain key 'pred'"
            pred = outputs["pred"]
        elif isinstance(outputs, Prediction):
            pred = outputs
        elif not batch:
            return
        else:
            raise TypeError(f"unpack_dict expects `outputs` to be dict or `Prediction`, got {type(outputs)}")
        assert isinstance(pred, Prediction)
        return wrapped(self, trainer, pl_module, pred, batch, batch_idx, *args, **kwargs)

    return func


class LoggingCallback(Callback, ABC, Generic[I, O]):
    r"""Callback that implements a limited size priority queue for items seen during an epoch.
    Only the top-k highest priority items from the epoch are retained. All items in the queue are
    logged at epoch completion.

    Args:
        name:
            Name / tag under which to log. State information will be prepended to ``name``.

        modes:
            Specific modes for wich this callback should run
    """

    def __init__(self, name: str, modes: ModeGroup = ALL_MODES):
        super().__init__()
        self.modes = tuple(Mode.from_group(modes))
        self.name = name

    @abstractmethod
    def __len__(self) -> int:
        r"""Returns the number of items pending logging."""
        ...

    @abstractmethod
    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        r"""Reset the state of this logging callback"""
        ...

    @abstractmethod
    def register(
        self,
        state: State,
        pl_module: BaseModel,
        example: I,
        prediction: O,
    ) -> None:
        r"""Performs any setup/registration needed for a given state. This method will only be
        called if ``state.mode in self.modes``. It may be called multiple times for a given state.
        """
        ...

    @abstractmethod
    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        r"""Handles callback logic when batch ends."""

    @abstractmethod
    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        r"""Handles callback logic when epoch ends."""
        ...

    def log_target(
        self,
        target: Any,
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        r"""Log an arbitrary target. This will probably be subclassed to support logging
        of whatever your callback generates.

        .. note:
            Do not attempt to override :class:`Callback.log` - Pytorch Lightning seems to
            have problems with this.
        """
        target_dict = {"trainer/global_step": step, tag: target}
        pl_module.logger.experiment.log(target_dict, commit=False)

    @unpack_dict
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if state.mode not in self.modes:
            return
        if not isinstance(outputs, Prediction):
            raise TypeError(f"Expected `outputs` to be type `Prediction`, found {type(outputs)}")
        if not isinstance(batch, Example):
            raise TypeError(f"Expected `batch` to be type `Example`, found {type(batch)}")
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    @unpack_dict
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if state.mode not in self.modes:
            return
        if not isinstance(outputs, Prediction):
            raise TypeError(f"Expected `outputs` to be type `Prediction`, found {type(outputs)}")
        if not isinstance(batch, Example):
            raise TypeError(f"Expected `batch` to be type `Example`, found {type(batch)}")
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    @unpack_dict
    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        state = pl_module.state
        if state.mode not in self.modes:
            return
        if not isinstance(outputs, Prediction):
            raise TypeError(f"Expected `outputs` to be type `Prediction`, found {type(outputs)}")
        if not isinstance(batch, Example):
            raise TypeError(f"Expected `batch` to be type `Example`, found {type(batch)}")
        self.register(state, pl_module, batch, outputs)
        self._on_batch_end(trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs)

    def on_train_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.TRAIN])
        assert len(self) == 0, "No items should be pending logging after reset"

    def on_validation_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.VAL])
        assert len(self) == 0, "No items should be pending logging after reset"

    def on_test_epoch_begin(self, *args, **kwargs):
        self.reset(specific_modes=[Mode.TEST])
        assert len(self) == 0, "No items should be pending logging after reset"

    def on_train_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TRAIN)

    def on_validation_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.VAL)

    def on_test_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TEST)

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: BaseModel):
        pl_module.state = pl_module.state.set_sanity_checking(True)

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: BaseModel):
        pl_module.state = pl_module.state.set_sanity_checking(False)

    @rank_zero_only
    def wrapped_log(
        self,
        target: Any,
        pl_module: BaseModel,
        tag: str,
        step: int,
    ):
        r"""Wrapper that calls self.log only on rank zero and when not sanity checking"""
        assert isinstance(pl_module, BaseModel)
        assert isinstance(tag, str) and tag
        assert isinstance(step, int) and step >= 0
        if not pl_module.state.sanity_checking:
            self.log_target(target, pl_module, tag, step)


class QueuedLoggingCallback(LoggingCallback, ABC, Generic[I, O]):
    r"""Callback that implements a limited size priority queue for items seen during an epoch.
    Only the top-k highest priority items from the epoch are retained. All items in the queue are
    logged at epoch completion, or at an interval if desired.

    Args:
        name:
            Name / tag under which to log. State information will be prepended to ``name``.

        modes:
            Modes for which this callback should execute.

        queue_size:
            Size of the priority queue

        target_cls:
            Type of :class:`LoggingTarget` that this callback should create from :class:`Example`,
            :class:`Prediction` pairs.

        flush_interval:
            By default, items will be enqueued over the course of an epoch and logged when the epoch
            concludes. Specifying ``flush_interval`` will flush the priority queue every ``flush_interval`` steps.
            If a ``flush_interval`` is specified, items in the queue at the end of an epoch will be discarded.

        negate_priority:
            If ``True``, use the negation of the priority return by :func:`get_priority`. Use this to log only
            the bottom-k priority items.
    """

    def __init__(
        self,
        name: str,
        queue_size: int,
        modes: ModeGroup = ALL_MODES,
        flush_interval: int = 0,
        negate_priority: bool = False,
    ):
        super().__init__(name, modes)
        self.queue_size = queue_size
        self.flush_interval = flush_interval
        self.queues = QueueStateCollection()
        self.negate_priority = negate_priority

    @abstractclassmethod
    def get_priority(cls, example: I, pred: O) -> Optional[Union[int, float]]:
        r"""Compute a priority for an example/prediction pair. When logging with a finite
        sized priority queue, only the ``len(queue)`` highest priority images will be logged.
        Typically priority would be assigned based on some metric (loss, entropy, error, etc.).
        If ``None`` is returned, assume the item should not be queued.
        """
        ...

    @abstractmethod
    def prepare_target(self, example: I, pred: O) -> Any:
        ...

    def __len__(self) -> int:
        return len(self.queues)

    def register(self, state: State, *args, **kwargs) -> None:
        r"""Register a queue for a given state."""
        if state not in self.queues.states:
            self.queues.register(state, maxsize=self.queue_size)

    def reset(self, specific_states: Iterable[State] = [], specific_modes: Iterable[Mode] = []):
        r"""Reset the state of this logging callback"""
        self.queues.reset(
            specific_states=list(specific_states),
            specific_modes=list(specific_modes),
        )

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
        *args,
        **kwargs,
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = pl_module.state

        # try to put this batch into the queue
        self.enqueue(batch, outputs, self.queues.get_state(state))

        # if a flush interval was specified, check if we need to flush
        # TODO should we use batch_idx for checking against flush_interval?
        step = trainer.global_step
        if self.flush_interval and (step % self.flush_interval == 0) and step:
            self.flush_queues(pl_module, state.mode, step)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        # discard queue at epoch end when using a flush_interval
        if self.flush_interval and not pl_module.state.sanity_checking:
            self.reset(specific_modes=[mode])

        # otherwise flush and log the queue
        else:
            step = trainer.global_step
            self.flush_queues(pl_module, mode, step)

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: BaseModel):
        self.flush_queues(pl_module, pl_module.state.mode, trainer.global_step)
        super().on_sanity_check_end(trainer, pl_module)

    def flush_queues(self, pl_module: BaseModel, mode: Mode, step: int):
        # ensure we only flush queues for the currently ending state
        queues_to_flush: Dict[State, PriorityQueue] = {
            state: queue for state, queue in self.queues.as_dict().items() if state.mode == mode
        }

        # dequeue and log all targets
        for state, queue in queues_to_flush.items():
            tag = state.with_postfix(self.name)
            targets = [self.prepare_target(example, pred) for example, pred in self.dequeue_all(queue)]
            self.wrapped_log(targets, pl_module, tag, step)

    @torch.no_grad()
    def enqueue(self, example: I, pred: O, queue: PriorityQueue) -> bool:
        r"""Enqueue an example/prediction pair to a given queue"""
        assert isinstance(queue, PriorityQueue)
        assert example.is_batched == pred.is_batched
        assert len(example) == len(pred)

        if not example.has_label:
            return False

        # recurse on batched input
        if example.is_batched:
            success = False
            for e, p in zip(example, pred):  # type: ignore
                e: I
                p: O
                success = success or self.enqueue(e, p, queue)
            return success

        # TODO consider clone/detach of example/pred

        # possibly enqueue item, depending on queue capacity and priority
        priority = self.get_priority(example, pred)
        if priority is None:
            return False
        priority = priority if not self.negate_priority else -1 * priority
        item = PrioritizedItem(priority, (example.cpu().detach(), pred.cpu().detach()))
        if queue.full():
            other = queue.get()
            item = max(item, other)
            insertion = item is not other
        else:
            insertion = True

        queue.put(item)
        return insertion

    @torch.no_grad()
    def dequeue_all(self, queue: PriorityQueue) -> Iterator[Tuple[I, O]]:
        r"""Dequeue and iterate through all items in a queue."""
        while not queue.empty():
            item = queue.get()
            example, pred = item.item
            yield example, pred
