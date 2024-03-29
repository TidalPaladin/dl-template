#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict

import pytest
import torch
import wandb

from project.callbacks import ImageLoggingCallback
from project.structs import BinaryPrediction, Example, Mode, MultiClassPrediction, State
from tests.test_callbacks.base_callback import BaseCallbackTest


class TestImageLoggingCallback(BaseCallbackTest):
    def test_init(self):
        cb = ImageLoggingCallback("img", queue_size=16)

    @pytest.fixture
    def callback(self, modes):
        cb = ImageLoggingCallback("name", 16, modes)
        return cb

    @pytest.mark.parametrize(
        "label,score,exp",
        [
            pytest.param(None, 0.5, None),
            pytest.param(1.0, 0.0, 1.0),
            pytest.param(0.0, 1.0, 1.0),
            pytest.param(0.0, 0.0, 0.0),
            pytest.param(1.0, 1.0, 0.0),
            pytest.param(0.5, 0.5, 0.0),
        ],
    )
    def test_get_priority_binary(self, label, score, exp):
        eps = 1e-8
        logit = torch.tensor(score).logit(eps=eps)

        example = Example(img=torch.rand(3, 32, 32), label=torch.tensor(label).view(1) if label is not None else None)
        pred = BinaryPrediction(logits=logit.view(1))
        priority = ImageLoggingCallback.get_priority(example, pred)
        if exp is None:
            assert priority is None
        else:
            assert abs(priority - exp) <= eps

    @pytest.mark.parametrize(
        "label,logits,exp",
        [
            pytest.param(None, (0, 0), None),
            pytest.param(0, (0.0, 0.0), 0.5),
            pytest.param(1, (0.0, 0.0), 0.5),
            pytest.param(0, (-1.0, 1.0), 1 - 0.1192029),  # probs -> (0.1192029, 0.8807971)
            pytest.param(1, (-1.0, 1.0), 1 - 0.8807971),
        ],
    )
    def test_get_priority_multiclass(self, label, logits, exp):
        example = Example(img=torch.rand(3, 32, 32), label=torch.tensor(label).view(1) if label is not None else None)
        pred = MultiClassPrediction(logits=torch.tensor(logits))
        priority = ImageLoggingCallback.get_priority(example, pred)
        if exp is None:
            assert priority is None
        else:
            assert abs(priority - exp) <= 1e-4

    def test_get_priority_exception_on_batched(self):
        example = Example(
            img=torch.rand(4, 3, 32, 32),
            label=torch.rand(4, 1),
        )
        pred = BinaryPrediction(logits=torch.rand(4, 1))
        with pytest.raises(ValueError):
            ImageLoggingCallback.get_priority(example, pred)

    @pytest.mark.parametrize(
        "max_size",
        [
            None,
            (16, 16),
            (32, 16),
        ],
    )
    def test_prepare_logging_target(self, max_size):
        example = Example(
            img=torch.rand(3, 32, 32),
            label=torch.rand(1),
        )
        pred = BinaryPrediction(logits=torch.rand(1))

        cb = ImageLoggingCallback("img", queue_size=16, max_size=max_size)
        result = cb.prepare_target(example, pred)

        assert isinstance(result, wandb.Image)

        if max_size is not None:
            H, W = example.img.shape[-2:]
            H_max, W_max = max_size
            H_out, W_out = result.image.height, result.image.width  # type: ignore
            assert H_out <= H_max
            assert W_out <= W_max
            assert H_out / W_out == H / W, "aspect ratio should be preserved"

    @pytest.mark.parametrize("queue_size", [4, 8])
    def test_enqueue(self, queue_size):
        eps = 1e-8
        example = Example(
            img=torch.rand(3, 32, 32),
            label=torch.tensor(1.0).view(1),
        )

        # build a dict of priority, prediction pairs
        total_size = 2 * queue_size
        priority = torch.rand(total_size)
        preds: Dict[float, BinaryPrediction] = {}
        for p in priority:
            logit = (1.0 - p).logit(eps=eps).view(1)
            pred = BinaryPrediction(logit)
            preds[p.item()] = pred

        # build another dict of top-k priority pairs
        keep_keys = priority.topk(k=queue_size).values
        keep_preds = {k.item(): preds[k.item()] for k in keep_keys}

        cb = ImageLoggingCallback("img", queue_size=queue_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)
        for k, p in preds.items():
            cb.enqueue(example, p, queue=queue)

        assert queue.qsize() <= queue_size
        queued_priorities = []
        while not queue.empty():
            item = queue.get()
            e, p = item.item
            assert isinstance(e, Example)
            assert isinstance(p, BinaryPrediction)
            queued_priorities.append(item.priority)

        out = torch.tensor(queued_priorities).sort(descending=True).values
        assert torch.allclose(out, keep_keys)

    def test_enqueue_null_priority(self):
        example = Example(
            img=torch.rand(3, 32, 32),
            label=None,
        )
        logit = torch.rand(1)
        pred = BinaryPrediction(logit)

        cb = ImageLoggingCallback("img", queue_size=8)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)

        assert cb.get_priority(example, pred) is None
        cb.enqueue(example, pred, queue=queue)
        assert queue.empty()

    def test_dequeue(self):
        eps = 1e-8
        example = Example(
            img=torch.rand(3, 32, 32),
            label=torch.tensor(1.0).view(1),
        )

        total_size = 8
        cb = ImageLoggingCallback("img", queue_size=total_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)

        # build a dict of priority, prediction pairs
        priority = torch.rand(total_size)
        preds: Dict[float, BinaryPrediction] = {}
        for p in priority:
            logit = (1.0 - p).logit(eps=eps).view(1)
            pred = BinaryPrediction(logit)
            preds[p] = pred
            cb.enqueue(example, pred, queue=queue)

        assert queue.qsize() == total_size
        dequeued = list(cb.dequeue_all(queue))

        assert queue.empty()
        assert len(dequeued) == total_size
        for e, p in dequeued:
            assert isinstance(e, Example)
            assert isinstance(p, BinaryPrediction)

    @pytest.mark.skip
    @pytest.mark.parametrize(
        "mode,should_log",
        [
            pytest.param(Mode.TRAIN, True),
            pytest.param(Mode.VAL, False),
            pytest.param(Mode.TEST, False),
        ],
    )
    def test_training_log(self, lightning_module, logger, mode, should_log):
        state = State(mode)
        lightning_module.state = state
        cb = ImageLoggingCallback("img", 8, modes=[mode])

        B = 4
        example = Example(img=torch.rand(B, 3, 32, 32), label=torch.randint(0, 1, (B, 1)))
        pred = BinaryPrediction(torch.rand(B, 1))

        cb.on_train_batch_end(
            lightning_module.trainer,
            lightning_module,
            pred,
            example,
            0,
        )

        if should_log:
            logger.experiment.log.assert_called()
            assert logger.experiment.log.call_count == B
        else:
            logger.experiment.log.assert_not_called()

    @pytest.mark.parametrize("mode", [Mode.VAL, Mode.TEST])
    @pytest.mark.parametrize("queue_size", [4, 8])
    def test_queued_log(self, lightning_module, logger, mode, queue_size):
        state = State(mode)
        lightning_module.state = state
        cb = ImageLoggingCallback("img", queue_size)

        B = 4
        for _ in range(3):
            example = Example(img=torch.rand(B, 3, 32, 32), label=torch.randint(0, 1, (B, 1)))
            pred = BinaryPrediction(torch.rand(B, 1))
            cb.on_train_batch_end(
                lightning_module.trainer,
                lightning_module,
                pred,
                example,
                0,
            )

        logger.experiment.log.assert_not_called()
        assert len(cb) <= queue_size

        cb._on_epoch_end(lightning_module.trainer, lightning_module, mode)

        logger.experiment.log.assert_called()
        assert 0 < logger.experiment.log.call_count <= queue_size

    def test_cpu_detach_on_enqueue(self, cuda):
        queue_size = 32
        device = "cuda:0" if cuda else "cpu"

        example = Example(
            img=torch.rand(3, 32, 32, device=device),
            label=torch.tensor(1.0, device=device).view(1),
        )
        pred = BinaryPrediction(torch.rand(1, requires_grad=True, device=device))

        cb = ImageLoggingCallback("img", queue_size=queue_size)
        state = State(Mode.TEST)
        cb.register(state)
        queue = cb.queues.get_state(state)
        cb.enqueue(example, pred, queue=queue)

        while not queue.empty():
            item = queue.get()
            e, p = item.item
            assert e.device == torch.device("cpu")
            assert p.device == torch.device("cpu")
            assert not e.requires_grad
            assert not p.requires_grad
