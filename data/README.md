# Saved experiment results

Raw results from real quantum hardware runs, committed so that
[`../notebooks/03_hardware.ipynb`](../notebooks/03_hardware.ipynb) renders for anyone
without an IBM Quantum account, and so the results stay reproducible after a free
account's allowance expires.

| file | contents |
|---|---|
| `hardware_qpe_t_gate.json` | QPE of the T gate (θ = 1/8), 3 estimation qubits. *Not yet produced; see below.* |

To generate it: put credentials in `.env` (see [`../.env.example`](../.env.example)), open
`03_hardware.ipynb`, set `RUN_ON_HARDWARE = True` and execute. The notebook writes the file
itself, recording the backend name, date, shot count and transpiled depth alongside the
counts.
