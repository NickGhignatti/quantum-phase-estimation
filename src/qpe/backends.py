"""One uniform way to run a circuit, whatever it is running on.

Notebooks and tests should never care whether a circuit is executing on an ideal
simulator, a noise model cloned from a real device, or actual IBM hardware.  They call
:func:`get_backend` and :func:`run_circuit` and get counts back.

Three backend kinds
-------------------
``"aer"``
    Ideal, noiseless local simulation.  The default, and what the test suite uses.
``"fake:<name>"``
    Local simulation using a noise model, coupling map and basis gates cloned from a
    real IBM device (e.g. ``"fake:FakeManilaV2"``).  Needs no credentials, but shows
    realistic error behaviour, the honest dress rehearsal for a hardware run.
``"ibm"``
    Real hardware via Qiskit Runtime.  Requires credentials; see the module notes below.

Credentials
-----------
Qiskit Runtime moved to the ``ibm_quantum_platform`` channel; the older ``ibm_quantum``
channel and the ``IBMQ`` provider no longer exist.  Provide credentials either by saving
an account once::

    QiskitRuntimeService.save_account(
        token="<API key>",
        instance="<CRN>",
        channel="ibm_quantum_platform",
        set_as_default=True,
    )

or by putting ``IBM_QUANTUM_TOKEN`` in a ``.env`` file at the repository root
(git-ignored), which this module reads automatically.  ``IBM_QUANTUM_INSTANCE`` may be
set alongside it to pin a specific instance, but can be left blank, in which case the
account's accessible instances are selected from automatically.

Transpilation
-------------
Every path here transpiles to ISA form before running.  This is not optional bookkeeping:
Aer rejects the opaque custom gates that :func:`qpe.core.qpe_circuit` builds, and hardware
requires circuits expressed in the device basis.  Doing it in one place keeps the
simulator and hardware paths honest: what the tests exercise is what hardware runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

__all__ = [
    "get_backend",
    "run_circuit",
    "submit_circuit",
    "counts_from_job",
    "retrieve_job",
    "counts_to_probabilities",
    "split_counts_by_register",
    "BackendHandle",
]

DEFAULT_OPTIMIZATION_LEVEL = 1


@dataclass(frozen=True)
class BackendHandle:
    """A backend plus the metadata needed to run and report on it."""

    backend: Any
    kind: str
    name: str

    @property
    def is_hardware(self) -> bool:
        return self.kind == "ibm"


def get_backend(spec: str = "aer") -> BackendHandle:
    """Resolve a backend spec string to a :class:`BackendHandle`.

    Parameters
    ----------
    spec
        ``"aer"``, ``"fake:<FakeBackendName>"``, ``"ibm"`` (least-busy suitable device),
        or ``"ibm:<backend_name>"`` for a specific device.
    """
    if spec == "aer":
        from qiskit_aer import AerSimulator

        return BackendHandle(AerSimulator(), "aer", "aer_simulator")

    if spec.startswith("fake:"):
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider

        fake_name = spec.split(":", 1)[1]
        try:
            fake_cls = getattr(fake_provider, fake_name)
        except AttributeError as exc:
            available = [n for n in dir(fake_provider) if n.startswith("Fake")]
            raise ValueError(
                f"Unknown fake backend {fake_name!r}. Available include: {available[:10]}"
            ) from exc
        fake = fake_cls()
        return BackendHandle(AerSimulator.from_backend(fake), "fake", fake_name)

    if spec == "ibm" or spec.startswith("ibm:"):
        service = _runtime_service()
        if spec == "ibm":
            backend = service.least_busy(simulator=False, operational=True)
        else:
            backend = service.backend(spec.split(":", 1)[1])
        return BackendHandle(backend, "ibm", backend.name)

    raise ValueError(
        f"Unrecognised backend spec {spec!r}; expected aer, fake:<name>, or ibm[:<name>]"
    )


def _runtime_service():
    """Build a ``QiskitRuntimeService``, preferring a saved account, falling back to .env."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    load_dotenv()
    token = os.getenv("IBM_QUANTUM_TOKEN")
    # A blank line in .env yields "", which is not a valid instance.  Normalise it to
    # "auto", which asks the service to pick among the instances the account can reach,
    # the right behaviour for a single-instance free account and quieter than passing
    # None, which selects the same way but warns about it.
    instance = os.getenv("IBM_QUANTUM_INSTANCE") or "auto"

    if token:
        return QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=instance
        )
    # No .env, so rely on a previously saved account.
    return QiskitRuntimeService()


def run_circuit(
    circuit: QuantumCircuit,
    backend: BackendHandle | str = "aer",
    shots: int = 4096,
    *,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int | None = 1234,
    return_isa: bool = False,
) -> dict[str, int] | tuple[dict[str, int], QuantumCircuit]:
    """Transpile and run ``circuit``, returning measurement counts.

    Parameters
    ----------
    return_isa
        Also return the transpiled circuit.  Useful for reporting depth and gate counts,
        which is the interesting number when explaining a hardware result.
    """
    job, isa = submit_circuit(
        circuit,
        backend,
        shots,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    counts = counts_from_job(job)

    return (counts, isa) if return_isa else counts


def submit_circuit(
    circuit: QuantumCircuit,
    backend: BackendHandle | str = "aer",
    shots: int = 4096,
    *,
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    seed_transpiler: int | None = 1234,
) -> tuple[Any, QuantumCircuit]:
    """Transpile and submit ``circuit``, returning ``(job, isa_circuit)`` without waiting.

    Split out from :func:`run_circuit` because a hardware job can sit in the queue for
    hours.  Submitting and collecting separately means the job id can be written to disk
    the moment it exists, so the result survives the notebook kernel dying mid-queue and
    can be recovered later with :func:`retrieve_job`.

    On simulator backends the job completes immediately and the distinction is moot; the
    same two-step path is used so that both are exercised by the same code.
    """
    handle = get_backend(backend) if isinstance(backend, str) else backend

    pm = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=handle.backend,
        seed_transpiler=seed_transpiler,
    )
    isa = pm.run(circuit)

    if handle.is_hardware:
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=handle.backend)
    else:
        from qiskit_aer.primitives import SamplerV2

        sampler = SamplerV2.from_backend(handle.backend)

    return sampler.run([isa], shots=shots), isa


def counts_from_job(job: Any) -> dict[str, int]:
    """Block until ``job`` finishes and return its counts.

    Accepts jobs from either sampler, and jobs recovered by :func:`retrieve_job`.
    """
    return _extract_counts(job.result()[0])


def retrieve_job(job_id: str) -> Any:
    """Look up a previously submitted hardware job by id.

    The recovery path for a queued run: save the id at submission time, and this returns
    the job later, in a fresh process, ready for :func:`counts_from_job`.
    """
    return _runtime_service().job(job_id)


def _extract_counts(pub_result: Any) -> dict[str, int]:
    """Pull counts out of a SamplerV2 pub result, whatever the classical registers are called.

    SamplerV2 returns results keyed by classical register name (``qpe_circuit`` uses
    ``"phase"``), unlike the old ``Result.get_counts()`` which returned a single dict.

    For circuits with several classical registers (HHL measures three), the per-shot
    bitstrings are joined into a single space-separated key.

    **Ordering contract:** segments appear in ``pub_result.data`` key order, which is the
    order the classical registers were *added* to the circuit, deliberately not Qiskit's
    usual display convention, which prints registers in reverse. Callers must split on
    whitespace and read segments in register-add order. Getting this backwards silently
    swaps registers and produces plausible-looking nonsense, so :func:`split_counts_by_register`
    is provided to avoid hand-rolled positional parsing.
    """
    data = pub_result.data
    fields = list(data.keys())
    if not fields:
        raise ValueError("Result contains no classical registers; was the circuit measured?")
    if len(fields) == 1:
        return data[fields[0]].get_counts()

    arrays = [data[f] for f in fields]
    joined: dict[str, int] = {}
    for bits in zip(*(a.get_bitstrings() for a in arrays), strict=True):
        key = " ".join(bits)
        joined[key] = joined.get(key, 0) + 1
    return joined


def split_counts_by_register(
    counts: dict[str, int], register_names: list[str]
) -> list[tuple[dict[str, str], int]]:
    """Decode multi-register counts into ``({register_name: bits}, count)`` pairs.

    Removes the need for positional unpacking of the joined keys produced by
    :func:`_extract_counts`, which is easy to get backwards.

    Parameters
    ----------
    register_names
        Classical register names in the order they were added to the circuit, i.e.
        ``[c.name for c in circuit.cregs]``.
    """
    out = []
    for key, count in counts.items():
        segments = key.split()
        if len(segments) != len(register_names):
            raise ValueError(
                f"result key {key!r} has {len(segments)} segments but "
                f"{len(register_names)} registers were named"
            )
        out.append((dict(zip(register_names, segments, strict=True)), count))
    return out


def counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Normalise a counts dict to probabilities."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}
