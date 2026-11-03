"""Tests for CPU machine-family override (CLI, YAML, and ALE_CPU_MACHINE_FAMILY)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ale_run import cli
from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import gcloud
from ale_run.orchestration.config_loader import _build_per_snapshot_env


def test_machine_chain_default_and_overrides() -> None:
    # Default (None) tries C-family first, then falls back to N2.
    assert gcloud._machine_chain("c4-standard-4", is_gpu=False) == (
        "c4-standard-4",
        "n2-standard-4",
    )
    assert gcloud._machine_chain("c4-highmem-4", is_gpu=False) == (
        "c4-highmem-4",
        "n2-highmem-4",
    )
    assert gcloud._machine_chain("n2-standard-8", is_gpu=False) == ("n2-standard-8",)

    # N2 override skips C-family completely while keeping non-C and GPU shapes intact.
    assert gcloud._machine_chain(
        "c4-standard-4", is_gpu=False, cpu_machine_family="n2"
    ) == ("n2-standard-4",)
    assert gcloud._machine_chain(
        "c4-highmem-4", is_gpu=False, cpu_machine_family="N2"
    ) == ("n2-highmem-4",)
    assert gcloud._machine_chain(
        "n2-standard-16", is_gpu=False, cpu_machine_family="n2"
    ) == ("n2-standard-16",)
    assert gcloud._machine_chain(
        "g2-standard-8", is_gpu=True, cpu_machine_family="n2"
    ) == ("g2-standard-8",)
    assert gcloud._machine_chain(
        "c4-standard-4", is_gpu=True, cpu_machine_family="n2"
    ) == ("c4-standard-4",)

    # C4 override pins C-family without N2 fallback.
    assert gcloud._machine_chain(
        "c4-standard-8", is_gpu=False, cpu_machine_family="c4"
    ) == ("c4-standard-8",)


@pytest.mark.parametrize("bad_family", ["e2", "t2d", "g2"])
def test_invalid_cpu_machine_family_rejected(bad_family: str) -> None:
    with pytest.raises(ValueError, match="cpu_machine_family must be one of"):
        gcloud.GcloudProvider(
            {
                "project": "test-project",
                "service_account_key": "unused.json",
                "cpu_machine_family": bad_family,
                "snapshots": {
                    "cpu-free-ubuntu": {
                        "image": "ale-ubuntu22-v1-1",
                        "zones": ["us-central1-a"],
                    }
                },
            }
        )


def test_acquire_with_n2_override_provisions_n2_and_pd_ssd(monkeypatch) -> None:
    monkeypatch.setenv("ALE_CPU_MACHINE_FAMILY", "n2")
    created_calls: list[tuple[str, ...]] = []

    async def fake_run_gcloud(*args: str, project: str, timeout: float | None = 1800.0):
        assert project == "test-project"
        if args[:3] == ("compute", "instances", "create"):
            created_calls.append(args)
            return (
                0,
                json.dumps([{"networkInterfaces": [{"networkIP": "10.0.0.5"}]}]),
                "",
            )
        raise AssertionError(f"unexpected gcloud call: {args}")

    monkeypatch.setattr(gcloud, "_run_gcloud", fake_run_gcloud)
    monkeypatch.setattr(gcloud, "wait_cua_ready", AsyncMock(return_value=True))

    provider = gcloud.GcloudProvider(
        {
            "project": "test-project",
            "service_account_key": "unused.json",
            "snapshots": {
                "cpu-free-ubuntu": {
                    "image": "ale-ubuntu22-v1-1",
                    "zones": ["us-central1-a"],
                },
                "gpu-free": {
                    "image": "ale-win10-v1-1",
                    "gpu": "nvidia-l4-vws",
                    "zones": ["us-central1-a"],
                },
            },
        }
    )

    cpu_handle = asyncio.run(
        provider.acquire(
            SandboxSpec(
                snapshot="cpu-free-ubuntu",
                os="linux",
                machine_type="c4-standard-4",
            )
        )
    )
    assert cpu_handle.metadata["machine_type"] == "n2-standard-4"
    assert len(created_calls) == 1
    assert "--machine-type=n2-standard-4" in created_calls[0]
    assert "--boot-disk-type=pd-ssd" in created_calls[0]

    gpu_handle = asyncio.run(
        provider.acquire(
            SandboxSpec(
                snapshot="gpu-free",
                os="windows",
                machine_type="g2-standard-8",
            )
        )
    )
    assert gpu_handle.metadata["machine_type"] == "g2-standard-8"
    assert "--machine-type=g2-standard-8" in created_calls[1]


def test_config_loader_and_cli_override_precedence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ALE_CPU_MACHINE_FAMILY", "c4")

    # 1. YAML cpu_machine_family overrides ALE_CPU_MACHINE_FAMILY env var.
    env_from_yaml = _build_per_snapshot_env(
        {
            "snapshots": {
                "cpu-free-ubuntu": {
                    "provider": "gcloud",
                    "image": "ale-ubuntu22-v1-1",
                    "gcloud": {
                        "project": "test-project",
                        "service_account_key": "unused.json",
                        "cpu_machine_family": "n2",
                        "zones": ["us-central1-a"],
                    },
                }
            }
        },
        "environment.yaml",
    )
    provider = gcloud.GcloudProvider(env_from_yaml.provider_specs["gcloud"].config)
    assert provider.config.cpu_machine_family == "n2"

    # 2. CLI --cpu-machine-family overrides both env var and YAML in _cmd_run.
    (tmp_path / "agent.yaml").write_text(
        "harness: dummy\nmodel: dummy-model\n", encoding="utf-8"
    )
    (tmp_path / "env.yaml").write_text(
        "snapshots:\n"
        "  cpu-free-ubuntu:\n"
        "    provider: gcloud\n"
        "    image: ale-ubuntu22-v1-1\n"
        "    gcloud:\n"
        "      project: test-project\n"
        "      service_account_key: unused.json\n"
        "      cpu_machine_family: c4\n"
        "      zones: [us-central1-a]\n",
        encoding="utf-8",
    )
    (tmp_path / "tasks.txt").write_text("demo/hello\n", encoding="utf-8")
    exp_yaml = tmp_path / "exp.yaml"
    exp_yaml.write_text(
        "name: test_exp\n"
        "agents: [agent.yaml]\n"
        "environment: env.yaml\n"
        "tasks: tasks.txt\n",
        encoding="utf-8",
    )

    captured_specs = []
    orig_runner = cli.Runner

    def spy_runner(spec):
        captured_specs.append(spec)
        return orig_runner(spec)

    monkeypatch.setattr(cli, "Runner", spy_runner)
    rc = cli.main(["run", str(exp_yaml), "--dry-run", "--cpu-machine-family", "n2"])
    assert rc == 0
    assert len(captured_specs) == 1
    assert (
        captured_specs[0].environment.provider_specs["gcloud"].config[
            "cpu_machine_family"
        ]
        == "n2"
    )
