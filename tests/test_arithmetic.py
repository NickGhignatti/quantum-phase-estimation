"""Exhaustive tests for the register-vs-register comparator.

This component is tested exhaustively and in isolation, deliberately: it is the one piece
of the HHL port with no reference implementation to check against, and a subtly wrong
comparator would not crash -- it would quietly produce a wrong final answer that still
looks like a plausible probability distribution.

Every input pair is checked for small widths, and crucially so is *ancilla cleanliness*:
if the scratch qubits did not return to |0> they would stay entangled with the operands
and corrupt the HHL uncomputation later.
"""

from __future__ import annotations

import itertools

import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from qpe.arithmetic import COMPARATOR_ANCILLA_QUBITS, comparator_ge_gate


def _run_comparison(a_val: int, b_val: int, n: int) -> tuple[int, int, int, bool]:
    """Prepare |a>|b>|0>|0...0>, apply the comparator, and decode the single basis state.

    Returns ``(a_out, b_out, result_bit, ancillas_clean)``.
    """
    n_anc = COMPARATOR_ANCILLA_QUBITS(n)
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    res = QuantumRegister(1, "res")
    anc = QuantumRegister(n_anc, "anc")
    qc = QuantumCircuit(a, b, res, anc)

    # Little-endian encoding: bit i of the integer goes on qubit i.
    for i in range(n):
        if (a_val >> i) & 1:
            qc.x(a[i])
        if (b_val >> i) & 1:
            qc.x(b[i])

    qc.append(comparator_ge_gate(n), [*a, *b, res[0], *anc])

    sv = Statevector(qc)
    probs = sv.probabilities_dict()
    # Basis inputs must give a single deterministic basis output.
    assert len(probs) == 1, f"expected a basis state, got {len(probs)} outcomes"
    bitstring = next(iter(probs))

    # Qiskit prints most-significant qubit first; reverse to index by qubit number.
    bits = bitstring[::-1]
    a_out = sum(int(bits[i]) << i for i in range(n))
    b_out = sum(int(bits[n + i]) << i for i in range(n))
    result = int(bits[2 * n])
    ancillas_clean = all(bits[2 * n + 1 + i] == "0" for i in range(n_anc))
    return a_out, b_out, result, ancillas_clean


@pytest.mark.parametrize("n", [1, 2, 3])
def test_comparator_exhaustive(n):
    """Check a >= b for every input pair, plus operand preservation and clean ancillas."""
    for a_val, b_val in itertools.product(range(2**n), repeat=2):
        a_out, b_out, result, clean = _run_comparison(a_val, b_val, n)

        assert result == int(a_val >= b_val), (
            f"n={n}: comparator said {a_val} >= {b_val} is {bool(result)}, "
            f"expected {a_val >= b_val}"
        )
        assert a_out == a_val, f"n={n}: operand a was modified ({a_val} -> {a_out})"
        assert b_out == b_val, f"n={n}: operand b was modified ({b_val} -> {b_out})"
        assert clean, f"n={n}: ancillas not restored to |0> for ({a_val}, {b_val})"


def test_comparator_in_superposition_stays_unentangled_with_ancillas():
    """With operands in superposition, ancillas must still factor out cleanly.

    The exhaustive basis-state test above cannot catch an ancilla that is only entangled
    for superposed inputs, which is exactly how HHL uses this gate.
    """
    n = 2
    n_anc = COMPARATOR_ANCILLA_QUBITS(n)
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    res = QuantumRegister(1, "res")
    anc = QuantumRegister(n_anc, "anc")
    qc = QuantumCircuit(a, b, res, anc)

    qc.h(a)  # uniform superposition over all a
    qc.x(b[0])  # b = 1
    qc.append(comparator_ge_gate(n), [*a, *b, res[0], *anc])

    sv = Statevector(qc)
    anc_indices = list(range(2 * n + 1, 2 * n + 1 + n_anc))
    anc_probs = sv.probabilities_dict(anc_indices)

    assert anc_probs.get("0" * n_anc, 0.0) == pytest.approx(1.0, abs=1e-9), (
        "ancillas remain entangled with the operands in superposition"
    )

    # a >= 1 holds for 3 of the 4 values of a, so the result bit is 1 with p=3/4.
    res_probs = sv.probabilities_dict([2 * n])
    assert res_probs.get("1", 0.0) == pytest.approx(0.75, abs=1e-9)


def test_rejects_invalid_width():
    with pytest.raises(ValueError):
        comparator_ge_gate(0)
