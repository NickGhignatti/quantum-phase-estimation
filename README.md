# Quantum Phase Estimation

## Project

- Read and study the Quantum Phase Estimation (QPE) protocol: https://pennylane.ai/qml/demos/tutorial_qpe

- Implement the algorithm in Qiskit and test it with a few examples to verify it works correctly. Then test it on a real quantum computer as well (e.g. IBMQ).

- Verify that the QPE algorithm is used within the HHL algorithm for solving linear systems: https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst#the-hhl-algorithm

## Goals

- [ ] **Study the Quantum Phase Estimation (QPE) protocol.**
  Read through the [PennyLane QPE tutorial](https://pennylane.ai/qml/demos/tutorial_qpe) to build a solid understanding of the algorithm before implementing it.

- [ ] **Implement QPE in Qiskit.**
  Write the algorithm and validate it against a handful of test cases (known eigenvalues/eigenvectors) to confirm it estimates phases correctly.

- [ ] **Run on real quantum hardware.**
  Execute the implementation on an actual quantum computer (e.g. via IBM Quantum) and compare the results against the simulator runs.

- [ ] **Connect QPE to HHL.**
  Verify how QPE is used as a subroutine within the HHL algorithm for solving linear systems of equations, using the [PennyLane HHL tutorial](https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst#the-hhl-algorithm) as a reference.
