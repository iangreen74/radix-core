"""Tests for the CLI interface."""

from typer.testing import CliRunner

from radix_core.cli.radix import cli

runner = CliRunner()


class TestCLIStatus:
    def test_status_runs(self):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Safety Configuration" in result.output

    def test_status_shows_dry_run(self):
        result = runner.invoke(cli, ["status"])
        assert "enabled" in result.output


class TestCLISubmit:
    def test_submit_basic(self):
        result = runner.invoke(cli, ["submit", "echo hello"])
        assert result.exit_code == 0
        assert "Job Submitted" in result.output
        assert "Job ID" in result.output

    def test_submit_with_options(self):
        result = runner.invoke(
            cli,
            [
                "submit",
                "echo test",
                "--name",
                "my-job",
                "--cpus",
                "2.0",
                "--memory",
                "1024",
                "--priority",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "my-job" in result.output

    def test_submit_shows_dry_run_status(self):
        result = runner.invoke(cli, ["submit", "echo hello"])
        assert result.exit_code == 0
        assert "Dry-run" in result.output


class TestCLIPlan:
    def test_plan_default(self):
        result = runner.invoke(cli, ["plan"])
        assert result.exit_code == 0
        assert "Execution Plan" in result.output

    def test_plan_with_count(self):
        result = runner.invoke(cli, ["plan", "--count", "5"])
        assert result.exit_code == 0

    def test_plan_json_output(self):
        result = runner.invoke(cli, ["plan", "--json"])
        assert result.exit_code == 0
        assert "plan_id" in result.output

    def test_plan_optimal(self):
        result = runner.invoke(cli, ["plan", "--planner", "optimal"])
        assert result.exit_code == 0


class TestCLIInfo:
    def test_info_runs(self):
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "Radix Core" in result.output
        assert "Available Components" in result.output

    def test_info_shows_components(self):
        result = runner.invoke(cli, ["info"])
        assert "FIFO Policy" in result.output
        assert "ThreadPool" in result.output


class TestCLIValidate:
    def test_validate_passes(self):
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_validate_checks_imports(self):
        result = runner.invoke(cli, ["validate"])
        assert "imports OK" in result.output
