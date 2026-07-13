import pennylane as qp
import numpy as np
import matplotlib.pyplot as plt


def U(wires):
    return qp.PhaseShift(2 * np.pi / 5, wires=wires)

dev = qp.device("default.qubit")

@qp.qnode(dev)
def circuit_qpe(estimation_wires):
    # initialize to state |1>
    qp.PauliX(wires=0)

    for wire in estimation_wires:
        qp.Hadamard(wires=wire)

    qp.ControlledSequence(U(wires=0), control=estimation_wires)

    qp.adjoint(qp.QFT)(wires=estimation_wires)

    return qp.probs(wires=estimation_wires)

estimation_wires = range(1, 5)

results = circuit_qpe(estimation_wires)

bit_strings = [f"0.{x:0{len(estimation_wires)}b}" for x in range(len(results))]

plt.bar(bit_strings, results)
plt.xlabel("phase")
plt.ylabel("probability")
plt.xticks(rotation="vertical")
plt.subplots_adjust(bottom=0.3)

plt.show()
