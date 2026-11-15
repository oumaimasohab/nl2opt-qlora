import json
from pathlib import Path

import pytest

from nl2opt.adapters import ResolvedAdapter, checkpoints, iteration_of
from nl2opt.tracking import find_run_id


@pytest.fixture
def adapter_dir(tmp_path):
    d = tmp_path / "v1"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"iters": 520, "project_name": "p"}))
    for name in ("0000100_adapters.safetensors", "0000500_adapters.safetensors", "adapters.safetensors"):
        (d / name).write_bytes(name.encode())
    (d / "wandb" / "run-20261115_115144-812vgqw8").mkdir(parents=True)
    (d / "wandb" / "run-20261115_120000-abcd1234").mkdir()
    return d


def test_checkpoints_sorted_with_final(adapter_dir):
    assert [(i, p.name) for i, p in checkpoints(adapter_dir)] == [
        (100, "0000100_adapters.safetensors"), (500, "0000500_adapters.safetensors"), (520, "adapters.safetensors")]


def test_iteration_of(adapter_dir):
    assert iteration_of(adapter_dir / "0000100_adapters.safetensors") == 100
    assert iteration_of(adapter_dir / "adapters.safetensors") == 520
    assert iteration_of(adapter_dir) == 520


def test_resolved_checkpoint_points_at_the_right_weights(adapter_dir):
    with ResolvedAdapter(adapter_dir / "0000100_adapters.safetensors") as resolved:
        resolved = Path(resolved)
        assert (resolved / "adapters.safetensors").read_bytes() == b"0000100_adapters.safetensors"
        assert json.loads((resolved / "adapter_config.json").read_text())["iters"] == 520
    assert not resolved.exists()


def test_resolved_directory_is_used_as_is(adapter_dir):
    with ResolvedAdapter(adapter_dir) as resolved:
        assert resolved == str(adapter_dir)


def test_find_run_id_takes_newest(adapter_dir, tmp_path):
    assert find_run_id(adapter_dir) == "abcd1234"
    assert find_run_id(tmp_path) is None
