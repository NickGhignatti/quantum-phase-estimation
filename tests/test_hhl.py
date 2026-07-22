"""Tests for the HHL port.

A note on test-case choice
--------------------------
The demo's own example, ``b = [1, 1]``, is a *weak* test: for
``A = [[3/8, 1/8], [1/8, 3/8]]`` that vector is exactly the ``lambda = 1/2`` eigenvector
``|+>``, so the solution is parallel to ``b`` and comes out right even if eigenvalue
inversion does nothing at all.  It is kept for fidelity to the reference, but the tests
that actually exercise the inversion use right-hand sides such as ``b = [1, 0]``, which
superpose both eigenvectors and therefore only give the correct answer if the two
eigenvalues are inverted with the correct *relative* weight.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qpe.hhl import (
    build_hhl_circuit,
    classical_solution,
    fake_inversion_gate,
    hamiltonian_evolution,
    solve_hhl,
)

A_DEMO = np.array([[3 / 8, 1 / 8], [1 / 8, 3 / 8]])
SHOTS = 120_000


def test_demo_matrix_has_power_of_two_eigenvalues():
    """The port's core assumption: eigenvalues must be of the form 2**-k."""
    eigs = np.linalg.eigvalsh(A_DEMO)
    assert np.allclose(sorted(eigs), [0.25, 0.5])


def test_evolution_time_makes_phases_dyadic():
    """t = -pi must map eigenvalues onto the QPE lattice exactly (theta = lambda/2).

    If this fails, every downstream result degrades silently rather than erroring.
    """
    from qiskit.quantum_info import Operator

    u = Operator(hamiltonian_evolution(A_DEMO)).data
    eigenvalues, eigenvectors = np.linalg.eigh(A_DEMO)

    for lam, vec in zip(eigenvalues, eigenvectors.T, strict=True):
        phase = np.angle((u @ vec) @ vec.conj()) % (2 * np.pi)
        theta = phase / (2 * np.pi)
        assert theta == pytest.approx(lam / 2, abs=1e-12)
        assert (theta * 2**3) == pytest.approx(round(theta * 2**3), abs=1e-12), (
            "theta is not representable with 3 estimation qubits"
        )


@pytest.mark.parametrize(("m_in", "expected"), [(1, 8), (2, 4), (4, 2)])
def test_fake_inversion_maps_theta_to_one_over_theta(m_in, expected):
    """theta = m/8 must be bit-reversed to the integer 1/theta.

    m=1 (theta=1/8) -> 8, m=2 (theta=1/4) -> 4, m=4 (theta=1/2) -> 2.
    """
    prec = 3
    phase = QuantumRegister(prec, "phase")
    inv = QuantumRegister(prec + 1, "inv")
    qc = QuantumCircuit(phase, inv)
    for i in range(prec):
        if (m_in >> i) & 1:
            qc.x(phase[i])
    qc.compose(fake_inversion_gate(prec), inplace=True)

    probs = Statevector(qc).probabilities_dict(list(range(prec, prec + prec + 1)))
    assert len(probs) == 1
    bits = next(iter(probs))[::-1]  # index by qubit number
    value = sum(int(bits[i]) << i for i in range(prec + 1))
    assert value == expected


@pytest.mark.parametrize(
    "b",
    [
        np.array([1.0, 1.0]),  # the demo's case (parallel to an eigenvector, weak)
        np.array([1.0, 0.0]),  # superposes both eigenvalues, exercises the inversion
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
    ],
)
def test_hhl_matches_classical_solution(b):
    """HHL must reproduce numpy's solution (in magnitude) to high fidelity."""
    exact = classical_solution(A_DEMO, b)
    result = solve_hhl(A_DEMO, b, precision=3, shots=SHOTS)

    assert result.fidelity_vs(exact) > 0.999, (
        f"b={b}: got {result.amplitudes}, expected |{exact}|"
    )
    assert result.success_rate > 0.0
    assert np.isclose(np.linalg.norm(result.amplitudes), 1.0)


def test_hhl_inverts_eigenvalues_with_correct_relative_weight():
    """The decisive test: unequal eigenvalue weighting must be reproduced.

    b = [1, 0] = (|+> + |->)/sqrt(2) has equal weight on both eigenvectors. Since
    lambda_+ = 1/2 and lambda_- = 1/4, the solution weights them 2:4, giving
    x proportional to [3, -1]. Any error in the relative inversion shows up here.
    """
    b = np.array([1.0, 0.0])
    result = solve_hhl(A_DEMO, b, precision=3, shots=SHOTS)

    expected = np.abs(np.array([3.0, -1.0]) / np.sqrt(10.0))
    assert result.amplitudes == pytest.approx(expected, abs=0.01)


def test_circuit_size_is_as_expected():
    """Guard against accidental qubit-count growth, which would break simulability."""
    qc = build_hhl_circuit(A_DEMO, np.array([1.0, 1.0]), precision=3)
    # sys 1 + phase 3 + inv 4 + case 4 + flag 1 + comparator ancillas 6
    assert qc.num_qubits == 19


def test_qpe_is_actually_used_as_a_subroutine():
    """Goal 4, asserted structurally: the HHL circuit must contain QPE and its inverse."""
    qc = build_hhl_circuit(A_DEMO, np.array([1.0, 1.0]), precision=3, measure=False)
    names = [instruction.operation.name for instruction in qc.data]
    qpe_uses = [n for n in names if "QPE" in n or "qpe" in n.lower()]
    assert len(qpe_uses) >= 2, f"expected QPE and its inverse, found {names}"


def test_rejects_invalid_input():
    with pytest.raises(ValueError, match="Hermitian"):
        build_hhl_circuit(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 1.0]))

    with pytest.raises(ValueError, match="power of two"):
        build_hhl_circuit(np.eye(3) * 0.5, np.array([1.0, 1.0, 1.0]))

    with pytest.raises(ValueError, match="does not match"):
        build_hhl_circuit(A_DEMO, np.array([1.0, 1.0, 1.0, 1.0]))
