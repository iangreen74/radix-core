"""Tests for configuration management."""

import pytest
from radix_core.config import (
    RadixConfig, SafetyConfig, ExecutionConfig, BatchingConfig, get_config, set_config, reset_config,
)


class TestSafetyConfig:
    def test_defaults_are_safe(self):
        cfg = SafetyConfig()
        assert cfg.dry_run is True
        assert cfg.no_deploy_mode is True
        assert cfg.cost_cap_usd == 0.0
        assert cfg.max_job_cost_usd == 0.0

    def test_dry_run_cannot_be_disabled(self):
        with pytest.raises(ValueError, match="DRY_RUN must be True"):
            SafetyConfig(dry_run=False)

    def test_cost_cap_must_be_zero(self):
        with pytest.raises(ValueError, match="COST_CAP_USD must be 0.00"):
            SafetyConfig(cost_cap_usd=1.0)

    def test_no_deploy_must_be_true(self):
        with pytest.raises(ValueError, match="NO_DEPLOY_MODE must be True"):
            SafetyConfig(no_deploy_mode=False)

    def test_is_frozen(self):
        cfg = SafetyConfig()
        with pytest.raises(Exception):
            cfg.dry_run = False


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.max_parallelism == 4
        assert cfg.default_executor == "local_subprocess"
        assert cfg.enable_gpu is False
        assert cfg.ray_local_mode is True

    def test_invalid_parallelism(self):
        with pytest.raises(ValueError, match="max_parallelism must be at least 1"):
            ExecutionConfig(max_parallelism=0)

    def test_invalid_executor(self):
        with pytest.raises(ValueError, match="Invalid executor"):
            ExecutionConfig(default_executor="kubernetes")

    def test_ray_must_be_local(self):
        with pytest.raises(ValueError, match="Ray must be in local mode"):
            ExecutionConfig(ray_local_mode=False)

    def test_gpu_without_enable_fails(self):
        with pytest.raises(ValueError, match="Cannot allocate GPUs"):
            ExecutionConfig(ray_num_gpus=1, enable_gpu=False)


class TestBatchingConfig:
    def test_defaults(self):
        cfg = BatchingConfig()
        assert cfg.default_batch_size == 32
        assert cfg.microbatch_size == 8

    def test_microbatch_cannot_exceed_batch(self):
        with pytest.raises(ValueError, match="microbatch_size cannot be larger"):
            BatchingConfig(default_batch_size=4, microbatch_size=8)


class TestRadixConfig:
    def test_defaults(self):
        cfg = RadixConfig()
        assert cfg.safety.dry_run is True
        assert cfg.execution.max_parallelism == 4

    def test_validate_returns_empty_on_valid(self):
        cfg = RadixConfig()
        assert cfg.validate() == []

    def test_to_dict(self):
        cfg = RadixConfig()
        d = cfg.to_dict()
        assert d["safety"]["dry_run"] is True
        assert "execution" in d
        assert "batching" in d

    def test_str(self):
        cfg = RadixConfig()
        s = str(cfg)
        assert "dry_run=True" in s
        assert "$0.00" in s


class TestGlobalConfig:
    def test_get_config_returns_default(self):
        cfg = get_config()
        assert isinstance(cfg, RadixConfig)

    def test_get_config_is_singleton(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_clears_singleton(self):
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b

    def test_set_config_validates(self):
        cfg = RadixConfig()
        set_config(cfg)
        assert get_config() is cfg
