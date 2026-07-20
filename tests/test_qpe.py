"""Tests for the QPE implementation.

The test cases are chosen so that assertions are *deterministic*: every phase in
``EXACT_PHASES`` is an exact multiple of ``2**-n``, so QPE concentrates all amplitude on
a single basis state and the measured outcome is that phase with probability 1.  This
turns "does QPE work?" into a hard assertion rather than a statistical one, and in
particular it catches bit-ordering bugs, which otherwise produce plausible-looking but
wrong distributions.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseGate, phase_estimation
from qiskit.quantum_info import Operator, Statevector

from qpe import qpe_circuit
from qpe.analysis import best_phase, counts_to_phases, expected_phase_distribution
from qpe.backends import run_circuit

SHOTS = 8192


@pytest.fixture
def eigenstate_of_phase_gate() -> QuantumCircuit:
    """|1> is the eigenstate of P(phi) with eigenvalue e^{i*phi}."""
    qc = QuantumCircuit(1)
    qc.x(0)
    return qc


# (label, theta, num_eval_qubits) -- every theta is an exact multiple of 2**-n
EXACT_PHASES = [
    ("T gate", 1 / 8, 3),
    ("S gate", 1 / 4, 3),
    ("Z gate", 1 / 2, 3),
    ("P(3pi/4)", 3 / 8, 3),
    ("5/16", 5 / 16, 4),
    ("7/16", 7 / 16, 4),
]


@pytest.mark.parametrize(("label", "theta", "n"), EXACT_PHASES)
def test_exact_dyadic_phase_is_certain(label, theta, n, eigenstate_of_phase_gate):
    """Dyadic phases must be recovered exactly, with probability ~1."""
    qc = qpe_circuit(PhaseGate(2 * np.pi * theta), n, eigenstate_of_phase_gate)
    counts = run_circuit(qc, "aer", shots=SHOTS)
    est = best_phase(counts, n)

    assert est.phase == pytest.approx(theta), f"{label}: wrong phase (likely a bit-ordering bug)"
    assert est.probability > 0.999, f"{label}: expected certainty, got p={est.probability}"
    assert est.error_vs(theta) == pytest.approx(0.0, abs=1e-12)


def test_pennylane_demo_case(eigenstate_of_phase_gate):
    """Reproduce the PennyLane tutorial: theta=0.2 with 4 qubits peaks at 0.1875.

    0.2 is not dyadic (0.001100110011... in binary), so QPE returns a distribution
    concentrated on the nearest representable values rather than a single answer.
    """
    theta, n = 0.2, 4
    qc = qpe_circuit(PhaseGate(2 * np.pi * theta), n, eigenstate_of_phase_gate)
    counts = run_circuit(qc, "aer", shots=SHOTS)
    est = best_phase(counts, n)

    assert est.phase == pytest.approx(0.1875), "should peak at the nearest 4-bit value"
    assert est.probability > 0.8, "the peak should still dominate"
    assert est.error_vs(theta) < est.resolution, "error must be within one resolution step"


def test_more_qubits_reduce_error(eigenstate_of_phase_gate):
    """Error on a non-dyadic phase must shrink as estimation qubits are added."""
    theta = 0.2
    errors = []
    for n in (2, 4, 6):
        qc = qpe_circuit(PhaseGate(2 * np.pi * theta), n, eigenstate_of_phase_gate)
        est = best_phase(run_circuit(qc, "aer", shots=SHOTS), n)
        errors.append(est.error_vs(theta))
    assert errors[0] > errors[1] > errors[2], f"error should decrease monotonically: {errors}"
    assert errors[-1] < 0.02


def test_superposition_input_yields_both_eigenphases():
    """QPE on a non-eigenstate samples eigenphases weighted by |amplitude|**2.

    With a phase gate, |0> has phase 0 and |1> has phase theta.  Preparing an equal
    superposition should therefore give roughly half the weight to each.
    """
    theta, n = 1 / 4, 3
    prep = QuantumCircuit(1)
    prep.h(0)

    qc = qpe_circuit(PhaseGate(2 * np.pi * theta), n, prep)
    dist = counts_to_phases(run_circuit(qc, "aer", shots=SHOTS), n)

    assert dist.get(0.0, 0) == pytest.approx(0.5, abs=0.05)
    assert dist.get(theta, 0) == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize(("label", "theta", "n"), EXACT_PHASES)
def test_matches_qiskit_builtin_phase_estimation(label, theta, n, eigenstate_of_phase_gate):
    """Our circuit must agree with Qiskit's built-in phase_estimation(), up to bit order.

    An independent cross-check of the physics. The two implementations use *opposite*
    bit-ordering conventions: ours makes estimation qubit 0 the least significant bit of
    the phase (see :mod:`qpe.qft`), whereas the built-in makes it the most significant,
    so its bitstrings read reversed. Both are correct QPE; only the readout convention
    differs, and ``test_exact_dyadic_phase_is_certain`` is what pins down that *ours* is
    the one that reads directly as a phase without further manipulation.

    Comparing the distributions after reversing the built-in's bits therefore checks the
    part that actually matters -- that we put the same amplitude on the same phase.
    """
    unitary = PhaseGate(2 * np.pi * theta)

    ours = qpe_circuit(unitary, n, eigenstate_of_phase_gate, measure=False)

    builtin = QuantumCircuit(n + 1)
    builtin.x(n)  # eigenstate |1> on the target, which builtin places last
    builtin.compose(phase_estimation(n, unitary), inplace=True)

    ours_probs = Statevector(ours).probabilities_dict(range(n))
    builtin_probs = {
        key[::-1]: p for key, p in Statevector(builtin).probabilities_dict(range(n)).items()
    }

    for key in set(ours_probs) | set(builtin_probs):
        assert ours_probs.get(key, 0.0) == pytest.approx(
            builtin_probs.get(key, 0.0), abs=1e-9
        ), f"{label}: disagreement with built-in on outcome {key!r}"

    # And ours reads as the true phase with no post-processing.
    assert max(ours_probs, key=ours_probs.__getitem__) == format(
        round(theta * 2**n), f"0{n}b"
    )


def test_analytic_distribution_matches_simulation(eigenstate_of_phase_gate):
    """The closed-form QPE distribution must match what the simulator produces."""
    theta, n = 0.3, 4
    qc = qpe_circuit(PhaseGate(2 * np.pi * theta), n, eigenstate_of_phase_gate)
    measured = counts_to_phases(run_circuit(qc, "aer", shots=SHOTS), n)
    theory = expected_phase_distribution(theta, n)

    for phase, p_theory in theory.items():
        assert measured.get(phase, 0.0) == pytest.approx(p_theory, abs=0.02)

    assert sum(theory.values()) == pytest.approx(1.0, abs=1e-9)


def test_controlled_powers_are_correct():
    """controlled_power(U, k) must equal a controlled U**k as a matrix."""
    from qpe.core import controlled_power

    unitary = PhaseGate(2 * np.pi * 0.3)
    for power in (1, 2, 4, 8):
        built = Operator(controlled_power(unitary, power))
        expected = Operator(PhaseGate(2 * np.pi * 0.3 * power).control(1))
        assert built.equiv(expected), f"controlled U^{power} is wrong"


def test_rejects_invalid_arguments(eigenstate_of_phase_gate):
    with pytest.raises(ValueError):
        qpe_circuit(PhaseGate(0.1), 0, eigenstate_of_phase_gate)

    two_qubit_prep = QuantumCircuit(2)
    with pytest.raises(ValueError, match="acts on"):
        qpe_circuit(PhaseGate(0.1), 3, two_qubit_prep)
