# Quantum Phase Estimation

Quantum Phase Estimation implemented from scratch in Qiskit, validated against exactly
known eigenvalues, run on real IBM Quantum hardware, and then reused as the subroutine at
the heart of the HHL algorithm for solving linear systems.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), a single self-contained installer, no prior Python setup needed.

```bash
uv sync --extra dev      # creates the environment and installs everything
uv run jupyter lab       # open the notebooks
```

That is the whole setup. To run the test suite:

```bash
uv run pytest
```

No IBM Quantum account is required for any of the above: the tests and three of the four
notebooks run on a local simulator. Credentials are only needed to re-run
`03_hardware.ipynb` against a real device, and even then, that notebook reads previously
saved results from `data/`, so it renders correctly without an account.

## Project goals

- [x] **Study the Quantum Phase Estimation (QPE) protocol.**
  [PennyLane QPE tutorial](https://pennylane.ai/qml/demos/tutorial_qpe) →
  derivation and worked reproduction in [`01_theory.ipynb`](notebooks/01_theory.ipynb).

- [x] **Implement QPE in Qiskit.**
  [`src/qpe/core.py`](src/qpe/core.py), validated in
  [`02_qpe_qiskit.ipynb`](notebooks/02_qpe_qiskit.ipynb) against known eigenvalues.

- [x] **Run on real quantum hardware.**
  [`03_hardware.ipynb`](notebooks/03_hardware.ipynb): ideal simulator → noise model
  cloned from a real device → IBM Quantum. Two experiments, each run on **`ibm_fez`** and
  **`ibm_marrakesh`** (156-qubit heavy-hex, depth 139 after transpilation, 4096 shots).
  Raw counts are committed in [`data/`](data/), so the notebook renders without an account.

  θ = 1/8 is exactly representable and both devices recover it, but with P = 0.80 on
  `ibm_marrakesh` against 0.54 on `ibm_fez`. θ = 0.2 is *not* representable, so success
  means preserving the ordering of the two straddling outcomes rather than a single peak —
  and there the devices diverge: `ibm_marrakesh` holds it, `ibm_fez` inverts it and returns
  the wrong modal estimate. The two also fail by different mechanisms, consistently across
  both experiments: `ibm_fez` leaks toward the all-zeros string (amplitude damping),
  `ibm_marrakesh` leaks to the most-significant-bit flip of its peak. Neither is predicted
  by the static noise model, and running on one device alone would have supported a
  conclusion that the other contradicts.

- [x] **Connect QPE to HHL.**
  [`src/qpe/hhl.py`](src/qpe/hhl.py) imports and calls `qpe_circuit` directly, ported
  from the [PennyLane/Qrisp HHL demo](https://pennylane.ai/qml/demos/linear_equations_hhl_qrisp_catalyst).
  Walkthrough in [`04_hhl.ipynb`](notebooks/04_hhl.ipynb).

## Layout

```
src/qpe/
  qft.py          inverse QFT + the bit-ordering convention the package obeys
  core.py         qpe_circuit(), the algorithm itself
  analysis.py     counts → phases, the analytic QPE distribution, plotting
  backends.py     one run() surface over simulator / noise model / real hardware
  arithmetic.py   register-vs-register comparator (Qiskit has none; HHL needs one)
  hhl.py          the HHL port, with QPE as its subroutine
tests/            pytest, deterministic assertions, exhaustive where feasible
notebooks/        the written report
```

## Notes on correctness

The test suite is built around cases whose answers are known *exactly*, so that
assertions are deterministic rather than statistical:

- **Dyadic phases.** For θ ∈ {1/8, 1/4, 3/8, 1/2, 5/16, 7/16}, QPE concentrates all
  amplitude on a single basis state, so the measured phase must be exact with
  probability 1. This is what pins down the bit-ordering convention: an endianness bug
  produces a plausible-looking but wrong distribution rather than an obvious failure.
- **Cross-check against Qiskit's built-in** `phase_estimation()`. The two use opposite
  bit-ordering conventions; the test compares them after accounting for that, which
  checks the physics while the dyadic tests pin down the readout.
- **Exhaustive comparator testing.** The register-vs-register comparator is checked on
  every input pair for small widths, including that its ancillas return to |0⟩, since leftover
  entanglement there would silently corrupt HHL's uncomputation.
- **HHL against numpy.** Note that the reference demo's own example, `b = [1, 1]`, is
  parallel to an eigenvector of `A` and so would pass even if eigenvalue inversion did
  nothing. The suite therefore also tests `b = [1, 0]`, which superposes both
  eigenvectors and only succeeds if the relative inversion weight is right.

## Deviations from the reference material

The HHL implementation is a deliberate port of the Qrisp/Catalyst demo, keeping its
simplifications (eigenvalues restricted to powers of two, `fake_inversion` in place of
general controlled rotations). Two things could not be translated directly:

| Qrisp | Here | Why |
|---|---|---|
| `@qrisp.RUS` repeat-until-success | post-selection on measured registers | Qiskit's equivalent needs mid-circuit measurement with feed-forward. Valid by the principle of deferred measurement: the uncomputation never touches the post-selected registers. The success rate is reported rather than hidden. |
| `case_indicator >= inv_res` | [`arithmetic.py`](src/qpe/arithmetic.py) | Qiskit's `IntegerComparator` compares against a *classical* constant only; HHL needs both operands quantum. |

## Environment

Pinned to Qiskit 2.x. Note that much of the Qiskit material online is out of date: the
`execute()` function and the `IBMQ` provider have been removed, and the `ibm_quantum`
channel has been replaced by `ibm_quantum_platform`. See [`.env.example`](.env.example)
for credential setup.
