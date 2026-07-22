"""Quantum arithmetic that Qiskit does not ship: a register-vs-register comparator.

Why this module exists
----------------------
The Qrisp HHL demo compares two *quantum* registers::

    qbl = (case_indicator >= inv_res)

Qiskit has no equivalent.  Its ``IntegerComparatorGate`` compares a register against a
**classical constant** only, which is not what HHL needs, because both operands are quantum and
in superposition.  So we build one.

How it works
------------
``a >= b`` is decided by the carry-out of the two's-complement subtraction ``a - b``,
computed as ``a + (~b) + 1``:

* flip ``b`` to ``~b`` with X gates,
* add with carry-in 1 using ``FullAdderGate``,
* the carry-out is 1 exactly when ``a - b >= 0``, i.e. when ``a >= b``.

Both operands must survive the comparison unchanged, because HHL keeps using them
afterwards, but ``FullAdderGate`` overwrites its second operand with the sum.  The
standard fix is **compute, copy, uncompute**: run the adder, CX the carry-out onto a
clean result qubit, then run the adder's inverse to restore everything.  Only the result
qubit is left correlated with the comparison; the sum and carry ancillas return to
``|0>`` and so do not pollute later interference.

Uncomputing is not optional.  Leftover entangled garbage would decohere the register the
HHL uncomputation step later relies on being clean.
"""

from __future__ import annotations

from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import FullAdderGate

__all__ = ["comparator_ge_gate", "COMPARATOR_ANCILLA_QUBITS", "comparator_qubit_layout"]


def COMPARATOR_ANCILLA_QUBITS(num_bits: int) -> int:  # noqa: N802 - reads as a constant
    """Number of ancilla qubits :func:`comparator_ge_gate` needs for ``num_bits`` operands.

    One carry-in, ``num_bits`` scratch qubits holding the running sum, one carry-out.
    """
    return num_bits + 2


def comparator_qubit_layout(num_bits: int) -> dict[str, list[int]]:
    """Index ranges of each logical register within the gate's qubit ordering.

    The gate acts on ``[a, b, result, ancilla]`` in that order.  Returned for callers
    (and tests) that would rather not recompute offsets by hand.
    """
    n = num_bits
    return {
        "a": list(range(0, n)),
        "b": list(range(n, 2 * n)),
        "result": [2 * n],
        "ancilla": list(range(2 * n + 1, 2 * n + 1 + COMPARATOR_ANCILLA_QUBITS(n))),
    }


def comparator_ge_gate(num_bits: int, *, label: str | None = None) -> Gate:
    """Return a gate flipping a result qubit iff ``a >= b``, for quantum ``a`` and ``b``.

    The gate acts on ``2 * num_bits + 1 + COMPARATOR_ANCILLA_QUBITS(num_bits)`` qubits,
    ordered ``[a (num_bits), b (num_bits), result (1), ancilla (num_bits + 2)]``.

    Both ``a`` and ``b`` are left unchanged, and all ancillas are returned to ``|0>``.
    The result qubit is XORed with the comparison outcome, so it must start in ``|0>``
    for the result to be the plain comparison bit.

    Registers are little-endian: ``a[0]`` is the least significant bit.
    """
    if num_bits < 1:
        raise ValueError(f"num_bits must be >= 1, got {num_bits}")

    n = num_bits
    a = QuantumRegister(n, "a")
    b = QuantumRegister(n, "b")
    res = QuantumRegister(1, "res")
    cin = QuantumRegister(1, "cin")
    scratch = QuantumRegister(n, "sum")
    cout = QuantumRegister(1, "cout")

    qc = QuantumCircuit(a, b, res, cin, scratch, cout, name=label or f"a>=b({n})")

    adder = FullAdderGate(n)
    # FullAdderGate maps |cin>|x>_n |y>_n |cout> with the sum landing in the y register.
    adder_qubits = [cin[0], *a, *scratch, cout[0]]

    def compute() -> None:
        # scratch <- ~b, so the adder computes a + ~b + cin without touching b itself.
        for i in range(n):
            qc.cx(b[i], scratch[i])
        qc.x(scratch)
        qc.x(cin)  # carry-in 1 completes the two's complement negation
        qc.append(adder, adder_qubits)

    compute()
    qc.cx(cout[0], res[0])  # copy out the answer before undoing the work
    qc.append(adder.inverse(), adder_qubits)
    qc.x(cin)
    qc.x(scratch)
    for i in range(n):
        qc.cx(b[i], scratch[i])

    return qc.to_gate()
