import pennylane as qp
import numpy as np

num_wires = 5
dev = qp.device("default.qubit", wires=num_wires)

# Quantum Lock
# It is represented by a unitary U which has all but one eigenvalue equal to 1, and the other equal to -1
# U|key> = -|key>
# But how can we differentiate the "key" eigenstate from the other eigenstate when the information is contained in the phase?
# That's where phase kickback comes in! When the correct eigenstate is input, the -1 phase imparted by U
# the unitary is kicked back to the ancilla, effectively changing its state from | - > to |+>

# In order to build our lock we can use FlipSign
def quantum_lock(secret_key):
    return qp.FlipSign(secret_key, wires=list(range(1, num_wires)))
# Next, we need to prepare the corresponding eigenstate for a key we want to try out.
# Remember, the lock is only unlocked by the "key" eigenstate with eigenvalue -1. We'll make use of BasisState to build the key:
def build_key(key):
    return qp.BasisState(key, wires=list(range(1, num_wires)))

@qp.set_shots(1)
@qp.qnode(dev)
def quantum_locking_mechanism(lock, key):
    build_key(key)
    qp.Hadamard(wires=0)  # Hadamard on ancilla qubit
    qp.ctrl(lock, control=0)  # Controlled unitary operation
    qp.Hadamard(wires=0)  # Hadamard again on ancilla qubit
    return qp.sample(wires=0)

def check_key(lock, key):
    if quantum_locking_mechanism(lock, key) == 1:
        print("Great job, you have uncovered the mysteries of the quantum universe!")
    else:
        print("Nice try, but that's not the right key!")


secret_key = np.array([0, 1, 1, 1])
lock = quantum_lock(secret_key)

check_key(lock, secret_key)
