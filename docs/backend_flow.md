# Backend function flow

This guide shows when the functions in `src/qpe/backends.py` and `src/qpe/runs.py` are called.

## 1. Normal QPE run

Used by the notebooks and tests when the caller wants counts immediately.

```text
qpe_circuit(...)
  (build the QPE circuit)
        |
        v
run_circuit(circuit, backend, shots)
  (public blocking entry point)
        |
        v
submit_circuit(circuit, backend, shots)
  (choose backend, transpile, and submit; do not wait)
        |
        +--> get_backend(spec)
        |      (turn "aer", "fake:...", or "ibm:..." into a BackendHandle)
        |            |
        |            +--> _runtime_service()
        |                   (only for IBM; load credentials and create the IBM service)
        |
        +--> generate_preset_pass_manager(...)
        |      (prepare a transpiler for the selected backend)
        |
        +--> pm.run(circuit)
        |      (rewrite the circuit into backend-supported ISA gates)
        |
        +--> SamplerV2.run([isa], shots=shots)
               (submit the transpiled circuit and return a job)
        |
        v
counts_from_job(job)
  (wait for the job and extract measurement counts)
        |
        v
_extract_counts(job.result()[0])
  (convert SamplerV2's result object into dict[str, int])
        |
        v
counts
  (for example: {"001": 4090, "000": 6})
        |
        v
best_phase(counts, n)
  (analysis.py: convert bitstrings into phases and choose the most likely one)
```

`run_circuit()` is therefore a wrapper around two lower-level operations:

```text
run_circuit()
  --> submit_circuit()
  --> counts_from_job()
```

If `return_isa=True`, `run_circuit()` returns both the counts and the transpiled circuit:

```python
counts, isa = run_circuit(circuit, "aer", return_isa=True)
```

The ISA circuit is useful for measuring the depth and gate count of the circuit that was
actually executed.

## 2. Choosing a backend

`get_backend()` is called inside `submit_circuit()` when the caller passes a string.

```text
submit_circuit(..., backend="aer")
        |
        v
get_backend("aer")
  (create an ideal local Aer simulator)
        |
        v
BackendHandle(backend, kind="aer", name="aer_simulator")
```

For a fake device:

```text
get_backend("fake:FakeManilaV2")
  (find the FakeManilaV2 class)
        |
        v
fake_cls()
  (create the fake IBM backend)
        |
        v
AerSimulator.from_backend(fake)
  (create a local simulator with the device's noise and connectivity)
        |
        v
BackendHandle(backend, kind="fake", name="FakeManilaV2")
```

For real hardware:

```text
get_backend("ibm:ibm_fez")
        |
        v
_runtime_service()
  (load IBM credentials from a saved account or .env)
        |
        v
service.backend("ibm_fez")
  (select the requested device)
        |
        v
BackendHandle(backend, kind="ibm", name="ibm_fez")
```

With just `"ibm"`, the code calls `service.least_busy(...)` instead and chooses an
operational non-simulator automatically.

## 3. Submit now, collect later

This path is used for hardware jobs that may wait in an IBM queue.

```text
submit_and_record(circuit, backend, pending_path)
  (runs.py: submit and immediately save the job ID)
        |
        v
submit_circuit(...)
  (backends.py: transpile and submit without waiting)
        |
        v
job.job_id()
  (obtain the identifier of the queued job)
        |
        v
pending_path.write_text(...)
  (save job ID, backend, shots, depth, and metadata)
```

Later, possibly in a new notebook session:

```text
collect_recorded(pending_path, result_path)
  (runs.py: read the saved job ID and collect the result)
        |
        v
json.loads(pending_path.read_text())
  (load the saved metadata)
        |
        v
retrieve_job(meta["job_id"])
  (backends.py: ask IBM for the old job object)
        |
        v
_runtime_service()
  (connect to IBM using the saved account or .env credentials)
        |
        v
counts_from_job(job)
  (wait for completion and extract counts)
        |
        v
result_path.write_text(...)
  (save metadata and counts permanently)
```

A saved result can later be read without contacting IBM:

```text
load_result(result_path)
  (read one saved JSON file, or return None if it does not exist)
```

To load all saved devices for one experiment:

```text
load_device_results(data_dir, experiment)
  (find matching hardware_qpe_... JSON files)
        |
        v
load_result(path)
  (read each file)
        |
        v
results keyed by backend name
```

## 4. Single-register result extraction

A normal QPE circuit has one classical register, usually called `phase`.

```text
job.result()[0]
        |
        v
_extract_counts(pub_result)
        |
        v
pub_result.data
  (find the classical result fields)
        |
        v
one field?
  |
  +--> yes: data[field].get_counts()
  |          (return the count dictionary directly)
  |
  +--> no: combine the bitstrings from all fields
```

The output is an ordinary dictionary that the analysis code can process:

```python
{"001": 4090, "000": 6}
```

If there are no classical registers, `_extract_counts()` raises an error because the
circuit probably was not measured.

## 5. Multiple-register result extraction for HHL

HHL measures three classical registers: `c_sys`, `c_case`, and `c_flag`.

```text
job.result()[0]
        |
        v
_extract_counts(pub_result)
  (get bitstrings for each classical register)
        |
        v
zip(register_bitstrings, strict=True)
  (match the results from the same shot)
        |
        v
"system_bits case_bits flag_bit"
  (join one shot into a single key)
        |
        v
counts
  (for example: {"01 000 0": 120})
        |
        v
split_counts_by_register(counts, ["c_sys", "c_case", "c_flag"])
  (replace positions with explicit register names)
        |
        v
{"c_sys": "01", "c_case": "000", "c_flag": "0"}
        |
        v
solve_hhl()
  (keep only the shots satisfying the HHL post-selection conditions)
```

`split_counts_by_register()` checks that every result key has exactly as many segments as
there are register names. This prevents silently assigning a bitstring to the wrong
register.

## 6. Probability conversion

This helper is independent of backend execution:

```text
counts_to_probabilities(counts)
  (divide every count by the total number of shots)
        |
        v
probabilities
```

For example:

```python
{"001": 600, "010": 300, "011": 100}
  -->
{"001": 0.6, "010": 0.3, "011": 0.1}
```

In ordinary QPE, `analysis.counts_to_phases()` performs a similar normalization while
also converting each measured bitstring into a phase such as `int("001", 2) / 2**n`.
