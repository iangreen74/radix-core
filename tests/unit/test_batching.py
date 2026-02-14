"""Tests for batching and microbatching."""

import time

from radix_core.batching.dynamic_batcher import (
    BatchingStrategy,
    BatchRequest,
    DynamicBatcher,
)
from radix_core.batching.microbatch import (
    EmbeddingTensorEstimator,
    MemoryEstimate,
    TextTensorEstimator,
)


class TestBatchRequest:
    def test_age_ms(self):
        req = BatchRequest(request_id="r1", data="hello", arrival_time=time.time() - 1.0)
        assert req.age_ms >= 900  # At least ~900ms old

    def test_is_expired(self):
        req = BatchRequest(
            request_id="r1",
            data="hello",
            arrival_time=time.time() - 2.0,
            max_latency_ms=1000,
        )
        assert req.is_expired is True

    def test_not_expired(self):
        req = BatchRequest(
            request_id="r1",
            data="hello",
            arrival_time=time.time(),
            max_latency_ms=60000,
        )
        assert req.is_expired is False

    def test_no_max_latency(self):
        req = BatchRequest(request_id="r1", data="hello", arrival_time=time.time() - 100)
        assert req.is_expired is False


class TestDynamicBatcher:
    def test_create_batcher(self):
        batcher = DynamicBatcher(processor=lambda batch: batch)
        assert batcher.max_batch_size > 0
        assert batcher.is_running is False

    def test_submit_request(self):
        batcher = DynamicBatcher(processor=lambda batch: batch)
        req = BatchRequest(request_id="r1", data="hello", arrival_time=time.time())
        req_id = batcher.submit_request(req)
        assert req_id == "r1"
        assert batcher.total_requests == 1

    def test_get_stats(self):
        batcher = DynamicBatcher(processor=lambda batch: batch)
        stats = batcher.get_stats()
        assert "total_requests" in stats
        assert stats["is_running"] is False

    def test_context_manager(self):
        batcher = DynamicBatcher(processor=lambda batch: batch)
        with batcher:
            assert batcher.is_running is True
        assert batcher.is_running is False

    def test_reset_stats(self):
        batcher = DynamicBatcher(processor=lambda batch: batch)
        req = BatchRequest(request_id="r1", data="x", arrival_time=time.time())
        batcher.submit_request(req)
        batcher.reset_stats()
        assert batcher.total_requests == 0


class TestTextTensorEstimator:
    def test_estimate_input_size(self):
        estimator = TextTensorEstimator()
        size = estimator.estimate_input_size("Hello, world!")
        assert size > 0

    def test_estimate_output_size(self):
        estimator = TextTensorEstimator()
        size = estimator.estimate_output_size("Hello, world!")
        assert size > 0

    def test_estimate_intermediate_size(self):
        estimator = TextTensorEstimator()
        size = estimator.estimate_intermediate_size("Hello, world!")
        assert size > 0

    def test_estimate_batch_memory(self):
        estimator = TextTensorEstimator()
        estimate = estimator.estimate_batch_memory(["hello", "world"])
        assert isinstance(estimate, MemoryEstimate)
        assert estimate.peak_size_mb > 0
        assert estimate.total_size_mb > 0


class TestEmbeddingTensorEstimator:
    def test_estimates(self):
        estimator = EmbeddingTensorEstimator(embedding_dim=768)
        assert estimator.estimate_input_size("hello") > 0
        assert estimator.estimate_output_size("hello") > 0
        assert estimator.estimate_intermediate_size("hello") > 0


class TestBatchingStrategy:
    def test_enum_values(self):
        assert BatchingStrategy.LATENCY_FIRST.value == "latency_first"
        assert BatchingStrategy.THROUGHPUT_FIRST.value == "throughput_first"
        assert BatchingStrategy.BALANCED.value == "balanced"
        assert BatchingStrategy.ADAPTIVE.value == "adaptive"
