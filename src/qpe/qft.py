"""Quantum Fourier Transform helpers and the bit-ordering convention.

Endianness is the single most common source of silently-wrong QPE results, so the
convention is fixed here, once, and everything else in the package obeys it.

Convention used throughout this package
---------------------------------------
Estimation register qubit ``j`` controls ``U^(2**j)``.  Consequently qubit 0 is the
*least* significant bit of the estimated phase, matching Qiskit's native little-endian
integer encoding.  A measured integer ``m`` on ``n`` estimation qubits therefore maps to

    theta = m / 2**n

with no bit reversal needed at readout, because Qiskit's ``QFTGate`` already includes
the swap network that undoes the QFT's natural bit reversal.

This means a Qiskit bitstring ``"001"`` (printed most-significant-first, so qubit 2,
qubit 1, qubit 0) is the integer 1, i.e. theta = 1/8 on three qubits.
"""

from __future__ import annotations

from qiskit.circuit import Gate
from qiskit.circuit.library import QFTGate


def inverse_qft_gate(num_qubits: int) -> Gate:
    """Return the inverse QFT as a gate on ``num_qubits`` qubits.

    ``QFTGate`` is preferred over the older circuit-based ``QFT`` class: it derives from
    ``Gate``, so the transpiler can reason about it abstractly and pick a synthesis
    strategy for the target backend instead of being handed a pre-expanded circuit.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}")
    return QFTGate(num_qubits).inverse()
