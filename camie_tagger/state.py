"""JSON run state: processed images, Tier 0 queue and Tier 0 progress."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .logging_setup import get_logger


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_within(path: Path, directories: list[Path]) -> bool:
    if not path.is_file():
        return False
    return any(path.is_relative_to(directory) for directory in directories)


class JsonFile:
    """A JSON document that is written atomically."""

    def __init__(self, path: Path, default: Any):
        self.path = path
        self._default = default
        self._data: Any | None = None

    @property
    def data(self) -> Any:
        if self._data is None:
            self._data = self._load()
        return self._data

    def replace(self, data: Any) -> None:
        self._data = data

    def _load(self) -> Any:
        if not self.path.exists():
            return json.loads(json.dumps(self._default))
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            get_logger().warning("Ignoring unreadable state file %s: %s", self.path, exc)
            return json.loads(json.dumps(self._default))

    def save(self) -> None:
        if self._data is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self._data, stream, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(temp_path, self.path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


class StateStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._processed = JsonFile(data_dir / "processed.json", {})
        self._queue = JsonFile(data_dir / "tier0_queue.json", [])
        self._progress = JsonFile(data_dir / "tier0_progress.json", {})

    # -- processed images -------------------------------------------------
    @property
    def processed(self) -> dict[str, dict]:
        return self._processed.data

    def is_processed(self, image_path: Path) -> bool:
        return str(image_path) in self.processed

    def mark_processed(self, image_path: Path, tag_count: int, result: str) -> None:
        self.processed[str(image_path)] = {
            "tagged_at": _utc_now(),
            "tags": tag_count,
            "result": result,
        }

    def replace_processed(self, records: dict[str, dict]) -> None:
        self._processed.replace(records)

    def save_processed(self) -> None:
        self._processed.save()

    # -- Tier 0 queue -----------------------------------------------------
    @property
    def tier0_queue(self) -> list[str]:
        return self._queue.data

    def enqueue_tier0(self, image_paths: list[Path]) -> int:
        queue = self.tier0_queue
        known = set(queue)
        added = 0
        for path in image_paths:
            key = str(path)
            if key not in known:
                queue.append(key)
                known.add(key)
                added += 1
        return added

    def save_queue(self) -> None:
        self._queue.save()

    def prune_queue(self, scan_dirs: list[Path]) -> int:
        """Drop queued images that are gone or no longer under the scan directories."""
        queue = self.tier0_queue
        kept = [item for item in queue if _is_within(Path(item), scan_dirs)]
        removed = len(queue) - len(kept)
        if removed:
            self._queue.replace(kept)
        return removed

    # -- Tier 0 progress --------------------------------------------------
    @property
    def tier0_progress(self) -> dict[str, str]:
        return self._progress.data

    def record_tier0(self, image_path: Path, status: str) -> None:
        self.tier0_progress[str(image_path)] = status

    def save_progress(self) -> None:
        self._progress.save()

    def save_all(self) -> None:
        self.save_processed()
        self.save_queue()
        self.save_progress()
