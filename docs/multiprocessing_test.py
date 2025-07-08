import time
import pennylane.numpy as np
import json
from qml_essentials.model import Model

min_n_samples = 500
max_n_samples = 5000
n_samples_step = 500
min_n_qubits = 2
max_n_qubits = 6
n_layers = 1
n_runs = 5

time_measure = time.time

print("Configuration:")
print(f"n_layers: {n_layers}")
print(f"min_n_samples: {min_n_samples}")
print(f"max_n_samples: {max_n_samples}")
print(f"n_samples_step: {n_samples_step}")
print(f"min_n_qubits: {min_n_qubits}")
print(f"max_n_qubits: {max_n_qubits}")
print(f"n_runs: {n_runs}")

results = {}

for run in range(n_runs):
    results[run] = {}
    for n_qubits in range(min_n_qubits, max_n_qubits + 1):
        results[run][n_qubits] = {}
        for n_samples in range(min_n_samples, max_n_samples + 1, n_samples_step):
            model = Model(
                n_qubits=n_qubits,
                n_layers=n_layers,
                circuit_type="Circuit_19",
                random_seed=1000,
            )
            model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)

            start = time_measure()
            model(execution_type="density")
            t_single = time_measure() - start

            model = Model(
                n_qubits=n_qubits,
                n_layers=n_layers,
                circuit_type="Circuit_19",
                mp_threshold=1000,
                random_seed=1000,
            )

            model.initialize_params(rng=np.random.default_rng(1000), repeat=n_samples)

            start = time_measure()
            model(execution_type="density")
            t_parallel = time_measure() - start

            print(
                f"{run} | {n_qubits} qubits | {n_samples} samples: {t_parallel / t_single}"
            )

            results[run][n_qubits][n_samples] = t_parallel / t_single

            break
        break
    break

with open("results.json", "w") as f:
    json.dump(results, f)
