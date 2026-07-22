"""Quantum Phase Estimation, built from scratch.

Given a unitary ``U`` and one of its eigenstates ``|psi>`` with

    U |psi> = exp(2*pi*i*theta) |psi>,

QPE writes an ``n``-bit approximation of ``theta`` into an estimation register.

The construction is the textbook one, kept deliberately explicit for teaching value:

1. Hadamard the estimation register into a uniform superposition.
2. For each estimation qubit ``j``, apply ``U`` controlled on that qubit, ``2**j`` times.
   Repeating the gate (rather than exponentiating the matrix) is the pedagogically honest
   version: it makes the exponential cost of QPE visible in the circuit itself, and it
   mirrors the reference implementations in the PennyLane and Qrisp tutorials.
3. Apply the inverse QFT to the estimation register.
4. Measure.

See :mod:`qpe.qft` for the bit-ordering convention, which the whole package obeys.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ClassicalRegister, Gate, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from .qft import inverse_qft_gate

__all__ = ["qpe_circuit", "controlled_power"]


def _as_gate(unitary: QuantumCircuit | Gate | Operator | np.ndarray) -> Gate:
    """Normalise any reasonable description of a unitary to a ``Gate``.

    Accepts a ``Gate``, a ``QuantumCircuit``, a ``quantum_info.Operator``, or a raw
    unitary matrix, the last two being the natural way to hand over a unitary obtained
    from classical linear algebra, which is how the multi-qubit tests supply one.
    """
    if isinstance(unitary, Gate):
        return unitary
    if isinstance(unitary, QuantumCircuit):
        return unitary.to_gate()
    if isinstance(unitary, Operator):
        return UnitaryGate(unitary.data)
    if isinstance(unitary, np.ndarray):
        return UnitaryGate(unitary)
    raise TypeError(
        f"Cannot interpret {type(unitary).__name__} as a unitary; expected Gate, "
        "QuantumCircuit, Operator or ndarray."
    )


def controlled_power(unitary: QuantumCircuit | Gate | Operator | np.ndarray, power: int) -> Gate:
    """Return a controlled-``U**power`` gate, built by repetition.

    ``power`` repetitions of controlled-``U`` are used rather than ``U.power(power)``
    followed by a single control.  Both are correct; repetition keeps the exponential
    gate count of QPE explicit in the circuit, which is the point being taught.
    """
    gate = _as_gate(unitary)
    n = gate.num_qubits

    block = QuantumCircuit(n, name=f"{gate.name}^{power}")
    for _ in range(power):
        block.append(gate, range(n))
    return block.to_gate().control(1)


def qpe_circuit(
    unitary: QuantumCircuit | Gate | Operator | np.ndarray,
    num_eval_qubits: int,
    state_prep: QuantumCircuit | Gate | None = None,
    *,
    measure: bool = True,
) -> QuantumCircuit:
    """Build a QPE circuit.

    Parameters
    ----------
    unitary
        The unitary ``U`` whose eigenphase is to be estimated.
    num_eval_qubits
        Number of estimation qubits ``n``.  The phase is resolved to a precision of
        ``2**-n``; phases that are exact multiples of ``2**-n`` are measured exactly,
        while others produce a distribution peaked at the nearest representable values.
    state_prep
        Circuit preparing the eigenstate ``|psi>`` on the target register.  If ``None``,
        the target register is left in ``|0...0>``, which is only meaningful when
        ``|0...0>`` happens to be an eigenstate of ``U``.
    measure
        If ``True``, append measurement of the estimation register into a classical
        register named ``"phase"``.  Set to ``False`` when the circuit is to be embedded
        as a subroutine (as in HHL, where it is later uncomputed).

    Returns
    -------
    QuantumCircuit
        Registers are ordered ``[eval, target]``, so estimation qubit ``j`` is qubit
        ``j`` of the circuit.
    """
    if num_eval_qubits < 1:
        raise ValueError(f"num_eval_qubits must be >= 1, got {num_eval_qubits}")

    gate = _as_gate(unitary)
    num_target = gate.num_qubits

    ev = QuantumRegister(num_eval_qubits, "eval")
    tgt = QuantumRegister(num_target, "target")
    qc = QuantumCircuit(ev, tgt, name="QPE")

    if state_prep is not None:
        prep = _as_gate(state_prep)
        if prep.num_qubits != num_target:
            raise ValueError(
                f"state_prep acts on {prep.num_qubits} qubits but the unitary acts on "
                f"{num_target}"
            )
        qc.append(prep, tgt)

    qc.h(ev)

    # Qubit j controls U^(2**j), which is what makes qubit 0 the least significant bit.
    for j in range(num_eval_qubits):
        qc.append(controlled_power(gate, 2**j), [ev[j], *tgt])

    qc.append(inverse_qft_gate(num_eval_qubits), ev)

    if measure:
        creg = ClassicalRegister(num_eval_qubits, "phase")
        qc.add_register(creg)
        qc.measure(ev, creg)

    return qc
