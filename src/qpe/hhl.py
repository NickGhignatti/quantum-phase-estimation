"""HHL for linear systems, ported from the Qrisp/Catalyst demo to Qiskit.

Reference: https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst

This is a faithful port: it keeps the demo's simplifications rather than building a
general HHL.  In particular eigenvalues must be of the form ``2**-k``, which is what makes
the ``fake_inversion`` bit-reversal trick a valid stand-in for real eigenvalue inversion.

**QPE is the subroutine at the heart of this file** -- ``qpe.core.qpe_circuit`` is imported
and used directly, first to write eigenvalues into a register and later, inverted, to
uncompute them.  That is the assignment's fourth goal demonstrated in code rather than prose.

Structure (mirroring the demo's four stages)
--------------------------------------------
1. Prepare ``|b>`` on the system register.
2. QPE with ``U = exp(-iAt)`` writes eigenvalue estimates ``theta_i = lambda_i * |t| / 2pi``.
3. ``fake_inversion`` bit-reverses ``theta`` into a register holding ``1/theta``.
4. A comparison against a Hadamard-conjugated "case indicator" register encodes ``1/lambda``
   into the *amplitude*; then stages 3 and 2 are uncomputed.

Two deviations from the Qrisp original, both deliberate
--------------------------------------------------------
``@qrisp.RUS`` -> post-selection
    Qrisp's repeat-until-success relies on Catalyst's dynamic control flow.  Qiskit's
    equivalent needs mid-circuit measurement with feed-forward, which is heavy and poorly
    supported on hardware.  Instead every register is measured at the end and the shots
    where ``case_indicator == 0`` and ``comparison == 0`` are kept.  This is legitimate by
    the *principle of deferred measurement*: the uncomputation acts only on the system,
    phase and inversion registers, never on the post-selected ones, so it commutes with
    those measurements and the final statistics are identical.  The price is that the
    discarded shots are wasted rather than retried -- so we report the success rate.

``case_indicator >= inv_res`` -> :func:`qpe.arithmetic.comparator_ge_gate`
    Qiskit has no register-vs-register comparator, so that module supplies one.

Why the comparison encodes ``1/lambda``
----------------------------------------
With the ``m``-qubit case indicator in uniform superposition over ``N = 2**m`` values and
``y`` the integer in the inversion register, the comparison marks the ``y`` values with
``c < y``.  Re-applying the Hadamards and projecting onto ``|0>`` gives amplitude
``y / N`` -- linear in ``y``, and ``y`` is proportional to ``1/lambda``.  That amplitude
proportional to ``1/lambda`` is precisely what HHL needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate, StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp

from .arithmetic import COMPARATOR_ANCILLA_QUBITS, comparator_ge_gate
from .core import qpe_circuit

__all__ = [
    "fake_inversion_gate",
    "hamiltonian_evolution",
    "build_hhl_circuit",
    "solve_hhl",
    "HHLResult",
    "classical_solution",
]

DEFAULT_EVOLUTION_TIME = -np.pi


def fake_inversion_gate(num_phase_qubits: int) -> QuantumCircuit:
    """Bit-reversal inversion of a phase register, as in the demo's ``fake_inversion``.

    Maps a phase ``theta = 2**-j`` held in ``num_phase_qubits`` qubits to the integer
    ``1/theta`` in a register of ``num_phase_qubits + 1`` qubits, by copying phase qubit
    ``i`` onto inversion qubit ``num_phase_qubits - i`` with a CX.

    This only works because eigenvalues are restricted to powers of two -- exactly the
    demo's simplifying assumption.  A general HHL would need controlled rotations here.

    Returns a circuit on ``[phase, inversion]``; the inversion register must start in
    ``|0...0>``.
    """
    n = num_phase_qubits
    phase = QuantumRegister(n, "phase")
    inv = QuantumRegister(n + 1, "inv")
    qc = QuantumCircuit(phase, inv, name="1/λ")
    for i in range(n):
        qc.cx(phase[i], inv[n - i])
    return qc


def hamiltonian_evolution(
    matrix: np.ndarray, time: float = DEFAULT_EVOLUTION_TIME, trotter_steps: int = 1
):
    """Build ``U = exp(-i * A * time)`` as a gate.

    ``time = -pi`` (the demo's choice) makes the QPE phases ``theta = lambda / 2``, so
    eigenvalues that are powers of two land exactly on the QPE lattice.

    For the 2x2 demo matrix the Pauli decomposition is ``0.375*I + 0.125*X``, whose terms
    commute, so Trotterization is *exact* and ``trotter_steps`` is irrelevant.  It matters
    only for the larger random matrices.
    """
    op = SparsePauliOp.from_operator(Operator(matrix))
    if trotter_steps == 1:
        return PauliEvolutionGate(op, time=time)
    from qiskit.synthesis import LieTrotter

    return PauliEvolutionGate(op, time=time, synthesis=LieTrotter(reps=trotter_steps))


@dataclass
class _Layout:
    """Qubit index bookkeeping for the HHL circuit."""

    n_sys: int
    precision: int

    @property
    def n_inv(self) -> int:
        return self.precision + 1

    @property
    def n_anc(self) -> int:
        return COMPARATOR_ANCILLA_QUBITS(self.n_inv)


def build_hhl_circuit(
    matrix: np.ndarray,
    b: np.ndarray,
    precision: int = 3,
    *,
    time: float = DEFAULT_EVOLUTION_TIME,
    trotter_steps: int = 1,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the HHL circuit for ``A x = b``.

    Parameters
    ----------
    matrix
        Hermitian ``A`` of size ``2**n x 2**n``, with eigenvalues of the form ``2**-k``.
    b
        Right-hand side; normalised internally.
    precision
        QPE estimation qubits.  Must be large enough to represent every ``lambda/2``
        exactly, or the inversion silently degrades.
    measure
        Append measurements of the system, case-indicator and comparison registers.

    Returns
    -------
    QuantumCircuit
        Classical registers, when present, are ``"c_sys"``, ``"c_case"`` and ``"c_flag"``.
    """
    dim = len(b)
    n_sys = int(np.log2(dim))
    if 2**n_sys != dim:
        raise ValueError(f"len(b) must be a power of two, got {dim}")
    if matrix.shape != (dim, dim):
        raise ValueError(f"matrix shape {matrix.shape} does not match len(b)={dim}")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("matrix must be Hermitian")

    lay = _Layout(n_sys, precision)

    sys = QuantumRegister(n_sys, "sys")
    phase = QuantumRegister(precision, "phase")
    inv = QuantumRegister(lay.n_inv, "inv")
    case = QuantumRegister(lay.n_inv, "case")
    flag = QuantumRegister(1, "flag")
    anc = QuantumRegister(lay.n_anc, "anc")
    qc = QuantumCircuit(sys, phase, inv, case, flag, anc, name="HHL")

    # --- Stage 1: prepare |b> -------------------------------------------------------
    b_vec = np.asarray(b, dtype=complex)
    b_vec = b_vec / np.linalg.norm(b_vec)
    qc.append(StatePreparation(list(b_vec)), sys)

    # --- Stage 2: QPE writes the eigenvalues ----------------------------------------
    # state_prep is None: |b> is already on the system register, and it must survive the
    # inverse QPE later rather than being undone by it.
    u_gate = hamiltonian_evolution(matrix, time=time, trotter_steps=trotter_steps)
    qpe_gate = qpe_circuit(u_gate, precision, state_prep=None, measure=False).to_gate()
    qc.append(qpe_gate, [*phase, *sys])

    # --- Stage 3: 1/lambda into a register ------------------------------------------
    inversion = fake_inversion_gate(precision)
    qc.append(inversion.to_gate(), [*phase, *inv])

    # --- Stage 4: 1/lambda into the amplitude ---------------------------------------
    comparator = comparator_ge_gate(lay.n_inv)
    qc.h(case)
    qc.append(comparator, [*case, *inv, flag[0], *anc])
    qc.h(case)

    # --- Uncompute stages 3 and 2 (reverse order, as Qrisp's `with invert()` does) ---
    qc.append(inversion.to_gate().inverse(), [*phase, *inv])
    qc.append(qpe_gate.inverse(), [*phase, *sys])

    if measure:
        from qiskit.circuit import ClassicalRegister

        # Names must differ from the quantum registers above -- Qiskit shares one namespace.
        c_sys = ClassicalRegister(n_sys, "c_sys")
        c_case = ClassicalRegister(lay.n_inv, "c_case")
        c_flag = ClassicalRegister(1, "c_flag")
        qc.add_register(c_sys, c_case, c_flag)
        qc.measure(sys, c_sys)
        qc.measure(case, c_case)
        qc.measure(flag, c_flag)

    return qc


@dataclass
class HHLResult:
    """Outcome of an HHL run."""

    amplitudes: np.ndarray
    """Normalised |x| recovered as sqrt(probability). Magnitudes only -- see note below."""
    probabilities: np.ndarray
    success_rate: float
    """Fraction of shots surviving post-selection."""
    shots: int
    kept_shots: int

    def fidelity_vs(self, exact: np.ndarray) -> float:
        """Overlap with a classical solution, comparing magnitudes.

        The readout recovers amplitudes as ``sqrt(probability)``, which discards sign and
        phase.  That is a property of the demo's measurement scheme, not a bug in the
        port, so comparison is against ``|exact|``.
        """
        ref = np.abs(np.asarray(exact, dtype=float))
        ref = ref / np.linalg.norm(ref)
        return float(np.abs(np.dot(self.amplitudes, ref)))


def classical_solution(matrix: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised classical solution of ``A x = b``, for validation."""
    x = np.linalg.solve(matrix, np.asarray(b, dtype=float))
    return x / np.linalg.norm(x)


def solve_hhl(
    matrix: np.ndarray,
    b: np.ndarray,
    precision: int = 3,
    *,
    backend: str = "aer",
    shots: int = 200_000,
    time: float = DEFAULT_EVOLUTION_TIME,
    trotter_steps: int = 1,
) -> HHLResult:
    """Run HHL and post-select to recover ``|x>``.

    Post-selection keeps only shots with ``case == 0`` and ``flag == 0``; see the module
    docstring for why this validly replaces Qrisp's repeat-until-success.
    """
    from .backends import run_circuit, split_counts_by_register

    dim = len(b)

    qc = build_hhl_circuit(
        matrix, b, precision, time=time, trotter_steps=trotter_steps, measure=True
    )
    counts = run_circuit(qc, backend, shots=shots)
    assert isinstance(counts, dict)

    # Decode by register *name* rather than position -- the joined-key segment order is
    # register-add order, which is the reverse of Qiskit's display convention and very
    # easy to get backwards.
    register_names = [creg.name for creg in qc.cregs]
    kept = np.zeros(dim)
    total = 0
    for registers, count in split_counts_by_register(counts, register_names):
        total += count
        # Post-selection: keep only the branch where the case indicator collapsed to 0
        # and the comparison flag is 0 (i.e. case < inv).
        if registers["c_flag"] != "0" or set(registers["c_case"]) != {"0"}:
            continue
        kept[int(registers["c_sys"], 2)] += count

    kept_total = kept.sum()
    if kept_total == 0:
        raise RuntimeError(
            "No shots survived post-selection; increase shots or check the eigenvalues "
            "are representable at this precision."
        )

    probs = kept / kept_total
    amps = np.sqrt(probs)
    return HHLResult(
        amplitudes=amps,
        probabilities=probs,
        success_rate=float(kept_total / total),
        shots=int(total),
        kept_shots=int(kept_total),
    )
