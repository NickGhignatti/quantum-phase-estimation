from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT


def build_qpe_circuit(unitary: Gate, num_counting_qubits: int) -> QuantumCircuit:
    """Standard QPE circuit for a single-qubit unitary with a known eigenstate prep.

    The eigenvector of `unitary` must already be prepared on the target qubit
    (the last qubit) before this circuit runs, e.g. by prepending an X gate
    for a unitary whose |1> eigenstate is the one being estimated.
    """
    target_qubit = num_counting_qubits
    qc = QuantumCircuit(num_counting_qubits + 1, num_counting_qubits)

    qc.h(range(num_counting_qubits))

    for counting_qubit in range(num_counting_qubits):
        repetitions = 2 ** counting_qubit
        controlled_u = unitary.control()
        for _ in range(repetitions):
            qc.append(controlled_u, [counting_qubit, target_qubit])

    qc.append(QFT(num_counting_qubits, inverse=True), range(num_counting_qubits))
    qc.measure(range(num_counting_qubits), range(num_counting_qubits))
    return qc
