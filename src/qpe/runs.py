"""Recording hardware runs to disk, so that a queued job survives the session.

A job submitted to IBM Quantum sits in a shared queue, sometimes for hours.  Two things
follow, and both are handled here rather than in the notebook:

*Submission and collection must be separable.*  If the only handle on a running job is a
blocking call in a live kernel, then closing the notebook loses a job that has already
consumed the account's allowance.  :func:`submit_and_record` writes the job id the moment
the job is accepted; :func:`collect_recorded` reads it back, in a later session if need
be, and only then waits.

*Results must outlive the account.*  The collected counts are written next to the id, and
:func:`load_result` reads them with no credentials at all, which is what lets
``03_hardware.ipynb`` render for a reader who has no IBM account, or after a free
allowance has expired.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit

from .backends import BackendHandle, counts_from_job, retrieve_job, submit_circuit

__all__ = ["submit_and_record", "collect_recorded", "load_result", "load_device_results"]


def submit_and_record(
    circuit: QuantumCircuit,
    backend: BackendHandle | str,
    pending_path: str | Path,
    *,
    shots: int = 4096,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, QuantumCircuit]:
    """Submit ``circuit`` and immediately write its job id to ``pending_path``.

    Returns ``(job, isa_circuit)`` without waiting for the result.  Anything in
    ``metadata`` (the true phase, the qubit count, whatever the notebook wants to report
    later) is stored alongside, so the collected file is self-describing.
    """
    handle = backend if isinstance(backend, BackendHandle) else None
    job, isa = submit_circuit(circuit, backend, shots=shots)

    pending_path = Path(pending_path)
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text(
        json.dumps(
            {
                "job_id": job.job_id(),
                "backend": handle.name if handle else str(backend),
                "submitted": datetime.datetime.now().isoformat(timespec="seconds"),
                "shots": shots,
                "transpiled_depth": isa.depth(),
                **(metadata or {}),
            },
            indent=2,
        )
    )
    return job, isa


def collect_recorded(
    pending_path: str | Path, result_path: str | Path
) -> dict[str, int]:
    """Fetch the job recorded at ``pending_path``, writing counts to ``result_path``.

    Blocks until the job finishes.  Safe to call from a fresh process: the job is located
    by the id on disk, not by any object held in memory.
    """
    pending_path, result_path = Path(pending_path), Path(result_path)
    meta = json.loads(pending_path.read_text())

    counts = counts_from_job(retrieve_job(meta["job_id"]))

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                **meta,
                "date": datetime.datetime.now().isoformat(timespec="seconds"),
                "counts": counts,
            },
            indent=2,
        )
    )
    return counts


def load_device_results(data_dir: str | Path, experiment: str) -> dict[str, dict[str, Any]]:
    """Every saved result for one experiment, keyed by device name, sorted by device.

    The same circuit run on several devices is the only way to tell a property of *QPE
    under noise* from a property of *one machine on one day*: a pattern that appears on
    one device and not another is the device's, not the algorithm's.

    Files are named ``hardware_qpe_<experiment>_<device>.json``; job-id files use a
    different prefix and so are not picked up here.
    """
    results = {}
    for path in sorted(Path(data_dir).glob(f"hardware_qpe_{experiment}_ibm_*.json")):
        loaded = load_result(path)
        if loaded is not None:
            results[loaded.get("backend", path.stem)] = loaded
    return dict(sorted(results.items()))


def load_result(result_path: str | Path) -> dict[str, Any] | None:
    """Read a saved hardware result, or ``None`` if it has not been produced yet.

    Needs no credentials, which is the point: it is how the notebook renders for a reader
    without an IBM account.
    """
    result_path = Path(result_path)
    if not result_path.exists():
        return None
    return json.loads(result_path.read_text())
