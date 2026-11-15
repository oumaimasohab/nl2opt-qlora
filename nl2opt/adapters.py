"""Locate LoRA adapters and their checkpoints, and resolve them to loadable directories.

mlx_lm.lora writes, in its adapter directory:
  adapter_config.json, adapters.safetensors (final), NNNNNNN_adapters.safetensors (every save_every
  steps), and wandb/run-<date>-<id>/ when report_to includes wandb.
mlx_lm.load(adapter_path=...) only accepts a directory holding adapter_config.json and
adapters.safetensors, so a checkpoint file is exposed through a temporary directory of symlinks.
"""

import json
import re
import tempfile
from pathlib import Path

CHECKPOINT = re.compile(r"^(\d{7})_adapters\.safetensors$")


def checkpoints(adapter_dir):
    """[(iteration, path)] for every saved checkpoint plus the final adapter, sorted by iteration."""
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    found = {int(m.group(1)): p for p in adapter_dir.iterdir() if (m := CHECKPOINT.match(p.name))}
    final = adapter_dir / "adapters.safetensors"
    if final.exists():
        found.setdefault(int(config["iters"]), final)
    return sorted(found.items())


def iteration_of(path):
    """Training iteration an adapter path corresponds to."""
    path = Path(path)
    if path.is_dir():
        return int(json.loads((path / "adapter_config.json").read_text())["iters"])
    if m := CHECKPOINT.match(path.name):
        return int(m.group(1))
    return int(json.loads((path.parent / "adapter_config.json").read_text())["iters"])


def adapter_root(path):
    path = Path(path)
    return path if path.is_dir() else path.parent


class ResolvedAdapter:
    """Context manager yielding a directory mlx_lm.load can use for a directory or checkpoint file."""

    def __init__(self, path):
        self.path = Path(path)
        self._tmp = None

    def __enter__(self):
        if self.path.is_dir():
            return str(self.path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._tmp = tempfile.TemporaryDirectory(prefix="nl2opt-adapter-")
        tmp = Path(self._tmp.name)
        (tmp / "adapter_config.json").symlink_to((self.path.parent / "adapter_config.json").resolve())
        (tmp / "adapters.safetensors").symlink_to(self.path.resolve())
        return str(tmp)

    def __exit__(self, *exc):
        if self._tmp:
            self._tmp.cleanup()
