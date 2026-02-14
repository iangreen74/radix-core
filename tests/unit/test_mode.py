"""Tests for execution mode management."""

from radix_core.mode import RadixMode, get_mode, is_production


class TestRadixMode:
    def test_default_is_development(self):
        assert get_mode() == RadixMode.DEVELOPMENT
        assert not is_production()

    def test_production_requires_explicit_env(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "production")
        assert get_mode() == RadixMode.PRODUCTION
        assert is_production()

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "PRODUCTION")
        assert is_production()

        monkeypatch.setenv("RADIX_MODE", "Production")
        assert is_production()

    def test_unknown_values_default_to_dev(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "staging")
        assert get_mode() == RadixMode.DEVELOPMENT
        assert not is_production()

    def test_empty_string_defaults_to_dev(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "")
        assert get_mode() == RadixMode.DEVELOPMENT

    def test_whitespace_stripped(self, monkeypatch):
        monkeypatch.setenv("RADIX_MODE", "  production  ")
        assert is_production()

    def test_enum_values(self):
        assert RadixMode.DEVELOPMENT.value == "development"
        assert RadixMode.PRODUCTION.value == "production"
