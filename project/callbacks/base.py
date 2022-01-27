#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from queue import PriorityQueue
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import pytorch_lightning as pl
import torch
import torchmetrics as tm
from pytorch_lightning.callbacks import Callback

from ..metrics import MetricStateCollection, PrioritizedItem, QueueStateCollection
from ..structs import Example, I, Mode, O, Prediction, State


T = TypeVar("T", bound="LoggingTarget")

if TYPE_CHECKING:
    from ..model.base import BaseModel
else:
    BaseModel = ForwardRef("BaseModel")

# Signature for Callback.on_X_batch_end
BatchEndCallable = Callable[[pl.Trainer, BaseModel, Prediction, Example, int, int], None]


def unpack_dict(wrapped) -> BatchEndCallable:
    r""":class:`LightningModule` is required to return either a loss tensor or a dictionary
    for automatic optimization. This decorator unpacks a returned dictionary by looking for a
    :class:`Prediction` instance under the key `pred`
    """

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

    return cast(BatchEndCallable, func)


@dataclass
class LoggingTarget(ABC, Generic[I, O]):
    r"""Callbacks that perform a logging operation produce :class:`LoggingTarget` instances.
    This decouples the operation of producing loggable objects from the logger-specific handling
    needed to log such objects.
    """

    @abstractmethod
    def log(
        self,
        pl_module: BaseModel,
        tag: str,
        step: int,
    ) -> Any:
        ...

    @classmethod
    def deferred_log(
        cls,
        pl_module: BaseModel,
        tag: str,
        step: int,
        targets: List[Any],
    ) -> None:
        r"""Some loggers require related items (such as a batch) to be grouped into a single log call.
        In such situations, ``log`` should return something to be logged for a single ``LoggingTarget``.
        Callbacks will aggregate the outputs of ``log`` and pass them to this function for deferred logging.
        """
        raise NotImplementedError(
            f"{cls.__name__}.deferred_log was called, but no implementation was provided. "
            "If you intend to defer logging, please provide an implementation. "
            "If you did not intend to defer logging, please ensure that `log` does not return anything."
        )

    @abstractclassmethod
    def create(cls: Type[T], example: I, pred: O) -> T:  # type: ignore
        ...


class LoggingCallback(Callback, Generic[I, O]):
    r"""Callback that implements a limited size priority queue for items seen during an epoch.
    Only the top-k highest priority items from the epoch are retained. All items in the queue are
    logged at epoch completion.

    Args:
        name:
            Name / tag under which to log. State information will be prepended to ``name``.

        queue_size:
            Size of the priority queue
    """

    def __init__(self, name: str, modes: Iterable[Mode], target_cls: Type[LoggingTarget]):
        super().__init__()
        self.modes = tuple(modes)
        self.name = name
        self.target_cls = target_cls

    @torch.no_grad()
    def prepare_logging_target(self, example: I, pred: O) -> LoggingTarget[I, O]:
        r"""Converts a raw example/prediction pair into an object to be logged"""
        return self.target_cls.create(example, pred)  # type: ignore

    def on_sanity_check_start(self, trainer: pl.Trainer, pl_module: BaseModel):
        pl_module.state = pl_module.state.set_sanity_checking(True)

    def on_sanity_check_end(self, trainer: pl.Trainer, pl_module: BaseModel):
        pl_module.state = pl_module.state.set_sanity_checking(False)


class QueuedLoggingCallback(LoggingCallback, Generic[I, O]):
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
        modes: Iterable[Mode],
        queue_size: int,
        target_cls: Type[LoggingTarget],
        flush_interval: int = 0,
        negate_priority: bool = False,
    ):
        super().__init__(name, modes, target_cls)
        self.queue_size = queue_size
        self.flush_interval = flush_interval
        self.queues = QueueStateCollection()
        self.negate_priority = negate_priority

    @abstractclassmethod
    def get_priority(cls, example: I, pred: O) -> Union[int, float]:
        r"""Compute a priority for an example/prediction pair. When logging with a finite
        sized priority queue, only the ``len(queue)`` highest priority images will be logged.
        Typically priority would be assigned based on some metric (loss, entropy, error, etc.).
        """
        ...

    def register(self, state: State) -> None:
        r"""Register a queue for a given state."""
        return self.queues.register(state, maxsize=self.queue_size)

    def clear_queues(self, mode: Optional[Mode] = None) -> None:
        r"""Register a queue for a given state."""
        if mode is None:
            self.queues.reset(specific_modes=self.modes)
        else:
            self.queues.reset(specific_modes=[mode])

    @unpack_dict
    def on_train_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    @unpack_dict
    def on_validation_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    @unpack_dict
    def on_test_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

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
        if not isinstance(outputs, Prediction):
            raise TypeError(f"Expected `outputs` to be type `Prediction`, found {type(outputs)}")
        if not isinstance(batch, Example):
            raise TypeError(f"Expected `batch` to be type `Example`, found {type(batch)}")

        state = self.state = pl_module.state
        if state.mode not in self.modes:
            return

        # register a queue for this state if needed
        if self.state not in self.queues.states:
            self.queues.register(self.state, maxsize=self.queue_size)

        # try to put this batch into the queue
        self.enqueue(batch, outputs, self.queues.get_state(state))

        # if a flush interval was specified, check if we need to flush
        # TODO should we use batch_idx for checking against flush_interval?
        step = trainer.global_step
        if self.flush_interval and (step % self.flush_interval == 0):
            self.flush_queues(pl_module, state.mode, step)

    def on_train_epoch_begin(self, *args, **kwargs):
        self._on_epoch_begin(*args, **kwargs, mode=Mode.TRAIN)

    def on_validation_epoch_begin(self, *args, **kwargs):
        self._on_epoch_begin(*args, **kwargs, mode=Mode.VAL)

    def on_test_epoch_begin(self, *args, **kwargs):
        self._on_epoch_begin(*args, **kwargs, mode=Mode.TEST)

    def on_train_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TRAIN)

    def on_validation_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.VAL)

    def on_test_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TEST)

    def _on_epoch_begin(
        self,
        *args,
        mode: Mode,
        **kwargs,
    ):
        self.clear_queues(mode)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        # discard queue at epoch end when using a flush_interval
        if self.flush_interval:
            self.clear_queues(mode)

        # otherwise flush and log the queue
        else:
            step = trainer.global_step
            self.flush_queues(pl_module, mode, step)

    def flush_queues(self, pl_module: BaseModel, mode: Mode, step: int):
        # ensure we only flush queues for the currently ending state
        queues_to_flush: Dict[State, PriorityQueue] = {
            state: queue for state, queue in self.queues.as_dict().items() if state.mode == mode
        }

        # dequeue and log all targets
        for state, queue in queues_to_flush.items():
            deferred: List[Any] = []
            tag = state.with_postfix(self.name)
            for (
                example,
                pred,
            ) in self.dequeue_all(queue):
                target = self.prepare_logging_target(example, pred)
                defer = target.log(pl_module, tag, step)
                if defer is not None:
                    deferred.append(defer)
            if deferred:
                self.target_cls.deferred_log(pl_module, tag, step, deferred)

    @property
    def total_queued_items(self) -> int:
        r"""Gets the total number of currently queued items across all states"""
        return len(self.queues)

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
        priority = priority if not self.negate_priority else -1 * priority
        item = PrioritizedItem(priority, (example, pred))
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


class IntervalLoggingCallback(LoggingCallback, Generic[I, O]):
    def __init__(self, name: str, modes: Iterable[Mode], log_interval: int, target_cls: Type[LoggingTarget]):
        super().__init__(name, modes, target_cls)
        self.log_interval = log_interval

    def on_train_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    def on_validation_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        outputs: O,
        batch: I,
        batch_idx: int,
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = self.state = pl_module.state
        needs_logging = state in self.modes and (batch_idx % self.log_interval == 0)
        if not needs_logging:
            return

        def log(e: I, p: O):
            target = self.prepare_logging_target(e, p)
            tag = state.with_postfix(self.name)
            target.log(pl_module, tag, trainer.global_step)

        if batch.is_batched:
            for e, p in zip(batch, outputs):
                log(e, p)  # type: ignore
        else:
            log(batch, outputs)


@dataclass
class MetricLoggingTarget(LoggingTarget[I, O]):
    metric: tm.MetricCollection

    @classmethod
    def create(
        cls: Type["MetricLoggingTarget"],
        metric: tm.MetricCollection,
        example: I,
        pred: O,
    ) -> "MetricLoggingTarget":
        metric.update(example, pred)
        return cls(metric)


# NOTE: metric.reset() must be explicitly called for Callback metrics
class MetricLoggingCallback(LoggingCallback, Generic[I, O]):
    target_cls: MetricLoggingTarget

    def __init__(
        self,
        name: str,
        modes: Iterable[Mode],
        collection: tm.MetricCollection,
        target_cls: Type[MetricLoggingTarget],
        log_on_step: bool = False,
    ):
        super().__init__(name, modes, target_cls)
        self.state_metrics = MetricStateCollection(collection)
        self.log_on_step = log_on_step

    @unpack_dict
    def on_train_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    @unpack_dict
    def on_validation_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    @unpack_dict
    def on_test_batch_end(self, *args, **kwargs):
        self._on_batch_end(*args, **kwargs)

    def _on_batch_end(
        self, trainer: pl.Trainer, pl_module: BaseModel, outputs: O, batch: I, batch_idx: int, *args, **kwargs
    ):
        r"""Since Callback.on_batch_end does not provide access to the batch and outputs, we must
        implement on_X_batch_end for each mode and call this method.
        """
        state = self.state = pl_module.state
        if state.mode not in self.modes:
            return

        self.state_metrics.register(state, device=torch.device(pl_module.device))
        collection = self.state_metrics.get_state(state)

        outputs = outputs.replace(logits=outputs.logits.float())
        target = self.target_cls.create(collection, batch, outputs)

        if self.log_on_step:
            tag = state.with_postfix(self.name)
            target.log(pl_module, tag, trainer.global_step)
            collection.reset()

    def on_train_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TRAIN)

    def on_validation_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.VAL)

    def on_test_epoch_end(self, *args, **kwargs):
        self._on_epoch_end(*args, **kwargs, mode=Mode.TEST)

    def _on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: BaseModel,
        mode: Mode,
    ):
        step = trainer.global_step
        for state, metric in self.state_metrics.as_dict().items():
            tag = state.with_postfix(self.name)
            if state.mode == mode:
                target = self.target_cls(metric)  # type: ignore
                target.log(pl_module, tag, step)
                metric.reset()
