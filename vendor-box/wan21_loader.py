"""Wan 2.1 image-to-video wrapper.

Generates a short ~3s video from a single still. Used after IDM-VTON to give
the snapshot photo subtle motion (breathing, slight pose shift) so the result
feels alive.

Contract:
    pipe = load_wan_i2v(path, device)
    mp4_bytes: bytes = pipe.run(still=PIL)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image


@dataclass
class WanI2VPipe:
    pipe: Any
    device: str

    def run(self, still: Image.Image) -> bytes:
        """Returns MP4 bytes (h264, ~3s, 720p)."""
        # TODO: integrate the real Wan 2.1 i2v pipeline. The official repo
        # ships an inference script that writes to disk; we want bytes in
        # memory so we wrap it.
        raise NotImplementedError("Wan 2.1 i2v pipeline not yet integrated")


def load_wan_i2v(path: Path, device: str = "cuda") -> WanI2VPipe:
    # TODO: load actual pipeline
    raise NotImplementedError(
        f"Loading Wan 2.1 from {path} not implemented — wire up the real loader on the vendor box"
    )
