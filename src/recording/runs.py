"""Versioned run directories with atomic snapshots and validated JSONL records."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .records import RunCategory

RUN_SCHEMA_VERSION = 2
_JSONL_NAMES = ("episodes", "updates", "evaluations")


class RunRecordingError(RuntimeError):
    """
    Report an invalid, inconsistent, or unsafe run-recording operation.
    """


class IncompleteRunError(RunRecordingError):
    """
    Report a run that was interrupted before a valid completion marker.
    """


@dataclass(frozen=True, slots=True)
class RunDirectory:
    """
    Own the fixed recorded-output layout for one isolated experimental run.

    Fields:
        * path: Root directory containing this run's files.
        * category: Experiment scope encoded in every recorded file.
        * run_id: Stable human-readable run identifier.
    """

    path: Path
    category: RunCategory
    run_id: str

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        category: RunCategory,
        run_id: str,
        manifest: dict[str, Any],
        config: dict[str, Any],
        metadata: dict[str, Any],
    ) -> RunDirectory:
        """
        Create an empty versioned run directory without overwriting an old run.
        """
        directory = Path(path)
        if directory.exists():
            if not directory.is_dir() or any(directory.iterdir()):
                raise RunRecordingError(
                    f"run directory already exists and is not empty: {directory}"
                )
        else:
            directory.mkdir(parents=True)
        run = cls(path=directory, category=category, run_id=run_id)
        run._write_snapshot("manifest.json", run._manifest(manifest))
        run._write_snapshot("config.json", run._document(config, "config"))
        run._write_snapshot("metadata.json", run._document(metadata, "metadata"))
        for name in _JSONL_NAMES:
            run._jsonl_path(name).touch(exist_ok=False)
        (directory / "checkpoints").mkdir(exist_ok=False)
        (directory / "trajectories").mkdir(exist_ok=False)
        return run

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_category: RunCategory | None = None,
        require_complete: bool = False,
    ) -> RunDirectory:
        """
        Open and validate a run, rejecting cross-category use and incomplete runs.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise RunRecordingError(f"run directory does not exist: {directory}")
        manifest = _read_json(directory / "manifest.json")
        _validate_document(manifest, "manifest")
        try:
            category = RunCategory(manifest["run_category"])
            run_id = manifest["run_id"]
        except (KeyError, ValueError) as error:
            raise RunRecordingError(
                "manifest has an invalid run category or run identifier."
            ) from error
        if not isinstance(run_id, str):
            raise RunRecordingError("manifest run_id must be a string.")
        if expected_category is not None and category is not expected_category:
            raise RunRecordingError(
                f"refusing to load {category.value} outputs as {expected_category.value}."
            )
        run = cls(path=directory, category=category, run_id=run_id)
        if require_complete:
            run.require_complete()
        run._validate_layout()
        return run

    def append(
        self, name: Literal["episodes", "updates", "evaluations"], record: Any
    ) -> None:
        """
        Append one checked, newline-delimited JSON metric record and flush it.
        """
        data = record.to_dict() if hasattr(record, "to_dict") else record
        if not isinstance(data, dict):
            raise RunRecordingError(
                "JSONL records must be dictionaries or expose to_dict()."
            )
        if data.get("schema_version") != RUN_SCHEMA_VERSION:
            raise RunRecordingError(
                "metric record schema version does not match the run schema."
            )
        record_category = data.get("run_category")
        if record_category != self.category.value:
            raise RunRecordingError(
                "metric record category does not match its run directory."
            )
        line = _canonical_json(data)
        with self._jsonl_path(name).open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{line}\n")
            output.flush()
            os.fsync(output.fileno())

    def records(
        self,
        name: Literal["episodes", "updates", "evaluations"],
    ) -> list[dict[str, Any]]:
        """
        Load fully valid JSONL records, rejecting partial/corrupt trailing writes.
        """
        contents = self._jsonl_path(name).read_text(encoding="utf-8")
        if contents and not contents.endswith("\n"):
            raise RunRecordingError(f"{name}.jsonl ends with a partial append.")
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(contents.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RunRecordingError(
                    f"invalid {name}.jsonl line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise RunRecordingError(
                    f"{name}.jsonl line {line_number} is not an object."
                )
            if record.get("schema_version") != RUN_SCHEMA_VERSION:
                raise RunRecordingError(
                    f"{name}.jsonl line {line_number} has an incompatible schema."
                )
            if record.get("run_category") != self.category.value:
                raise RunRecordingError(
                    f"{name}.jsonl line {line_number} has a mismatched category."
                )
            records.append(record)
        return records

    def write_trajectory(self, name: str, trajectory: dict[str, Any]) -> Path:
        """
        Atomically persist one selected evaluation trajectory beneath trajectories/.
        """
        destination = self.path / "trajectories" / f"{_safe_name(name)}.json"
        self._write_path(destination, self._document(trajectory, "trajectory"))
        return destination

    def complete(self, summary: dict[str, Any]) -> None:
        """
        Atomically mark the run complete after all preceding records are durable.
        """
        if (self.path / "completion.json").exists():
            raise RunRecordingError("a completed run cannot be completed again.")
        self._validate_layout()
        completion = self._document(summary, "completion")
        completion["state"] = "complete"
        completion["completed_at"] = _utc_now()
        self._write_snapshot("completion.json", completion)

    def require_complete(self) -> dict[str, Any]:
        """
        Return a valid completion record or report an interrupted run.
        """
        completion_path = self.path / "completion.json"
        if not completion_path.is_file():
            raise IncompleteRunError(f"run has no completion marker: {self.path}")
        try:
            completion = _read_json(completion_path)
            _validate_document(completion, "completion")
        except RunRecordingError as error:
            raise IncompleteRunError(
                f"run completion marker is invalid: {self.path}"
            ) from error
        if completion.get("state") != "complete":
            raise IncompleteRunError(
                f"run completion marker is not complete: {self.path}"
            )
        if completion.get("run_category") != self.category.value:
            raise IncompleteRunError(
                f"run completion category is inconsistent: {self.path}"
            )
        return completion

    def manifest_checksum(self) -> str:
        """
        Return the SHA-256 identity of the canonical manifest snapshot.
        """
        return _sha256(_canonical_json(_read_json(self.path / "manifest.json")))

    def _manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """
        Add immutable run identity fields to a caller-provided manifest.
        """
        document = self._document(manifest, "manifest")
        document.update(
            {
                "run_id": self.run_id,
                "run_category": self.category.value,
                "created_at": _utc_now(),
            }
        )
        return document

    def _document(self, contents: dict[str, Any], document_type: str) -> dict[str, Any]:
        """
        Build one versioned JSON snapshot with category identity.
        """
        if not isinstance(contents, dict):
            raise RunRecordingError(f"{document_type} must be a JSON object.")
        document = dict(contents)
        document.update(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "document_type": document_type,
                "run_category": self.category.value,
            }
        )
        _canonical_json(document)
        return document

    def _validate_layout(self) -> None:
        """
        Confirm that required snapshots, streams, and directories exist.
        """
        for snapshot, document_type in (
            ("config.json", "config"),
            ("metadata.json", "metadata"),
        ):
            document = _read_json(self.path / snapshot)
            _validate_document(document, document_type)
            if document.get("run_category") != self.category.value:
                raise RunRecordingError(
                    f"{snapshot} category does not match manifest category."
                )
        for name in _JSONL_NAMES:
            if not self._jsonl_path(name).is_file():
                raise RunRecordingError(f"missing required metric stream: {name}.jsonl")
            self.records(name)
        for directory in ("checkpoints", "trajectories"):
            if not (self.path / directory).is_dir():
                raise RunRecordingError(f"missing required run directory: {directory}")

    def _write_snapshot(self, name: str, document: dict[str, Any]) -> None:
        """
        Atomically write one root JSON document.
        """
        self._write_path(self.path / name, document)

    def _write_path(self, destination: Path, document: dict[str, Any]) -> None:
        """
        Replace a JSON snapshot atomically after syncing its temporary file.
        """
        payload = f"{_canonical_json(document)}\n"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _jsonl_path(self, name: str) -> Path:
        """
        Return one fixed JSONL path after rejecting arbitrary filenames.
        """
        if name not in _JSONL_NAMES:
            raise RunRecordingError(f"unsupported metric stream: {name}")
        return self.path / f"{name}.jsonl"


def collect_run_metadata(
    *,
    repository: str | Path = ".",
    device: str | None = None,
    environment_workers: int | None = None,
    intraop_threads: int | None = None,
    interop_threads: int | None = None,
) -> dict[str, Any]:
    """
    Collect reproducibility context while making unavailable hardware explicit.
    """
    repository_path = Path(repository)
    metadata: dict[str, Any] = {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "processor": platform.processor() or None,
        "git": _git_metadata(repository_path),
        "dependency_freeze": _dependency_freeze(),
        "execution": {
            "requested_device": device,
            "environment_workers": environment_workers,
            "intraop_threads": intraop_threads,
            "interop_threads": interop_threads,
        },
        "torch": _torch_metadata(device),
    }
    return metadata


def _git_metadata(repository: Path) -> dict[str, str | bool | None]:
    """
    Return commit identity and dirty state without assuming Git is installed.
    """
    commit = _run_command(["git", "rev-parse", "HEAD"], repository)
    status = _run_command(["git", "status", "--porcelain"], repository)
    return {
        "commit": commit or None,
        "dirty": None if status is None else bool(status),
    }


def _dependency_freeze() -> list[str] | None:
    """
    Return the active interpreter's dependency freeze when pip is available.
    """
    output = _run_command([sys.executable, "-m", "pip", "freeze"], Path.cwd())
    return None if output is None else output.splitlines()


def _torch_metadata(device: str | None) -> dict[str, Any]:
    """
    Return available PyTorch, CUDA, and GPU facts without requiring CUDA.
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "requested_device": device}
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    driver_version = (
        _run_command(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            Path.cwd(),
        )
        if cuda_available
        else None
    )
    return {
        "available": True,
        "version": str(torch.__version__),
        "requested_device": device,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "driver_version": driver_version.splitlines()[0] if driver_version else None,
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
    }


def _run_command(arguments: list[str], directory: Path) -> str | None:
    """
    Run a small metadata command and return stripped output or None on failure.
    """
    try:
        result = subprocess.run(
            arguments,
            cwd=directory,
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read_json(path: Path) -> dict[str, Any]:
    """
    Read one JSON object while converting malformed snapshots into a recording error.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunRecordingError(f"invalid JSON snapshot: {path}") from error
    if not isinstance(data, dict):
        raise RunRecordingError(f"JSON snapshot is not an object: {path}")
    return data


def _validate_document(document: dict[str, Any], document_type: str) -> None:
    """
    Check the common version and document-type fields of a JSON snapshot.
    """
    if document.get("schema_version") != RUN_SCHEMA_VERSION:
        raise RunRecordingError(f"{document_type} has an incompatible schema version.")
    if document.get("document_type") != document_type:
        raise RunRecordingError(f"expected a {document_type} document.")


def _canonical_json(data: Any) -> str:
    """
    Encode finite JSON deterministically for persistence and checksums.
    """
    try:
        return json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RunRecordingError(
            "run data is not finite JSON-compatible data."
        ) from error


def _sha256(text: str) -> str:
    """
    Return the SHA-256 hexadecimal digest of UTF-8 text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_name(name: str) -> str:
    """
    Validate a trajectory stem so writes cannot escape trajectories/.
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise RunRecordingError("trajectory name must be a non-empty file stem.")
    return name


def _utc_now() -> str:
    """
    Return an explicit UTC timestamp for run provenance.
    """
    return datetime.now(UTC).isoformat()
