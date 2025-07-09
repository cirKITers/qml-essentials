import time
import pennylane.numpy as np
import json
from qml_essentials.model import Model
import matplotlib.pyplot as plt

seed = 1000
min_n_samples = 500
max_n_samples = 8500
n_samples_step = 1000
min_n_qubits = 2
max_n_qubits = 6
n_layers = 1
n_runs = 8

time_measure = time.time

print("Configuration:")
print(f"n_layers: {n_layers}")
print(f"min_n_samples: {min_n_samples}")
print(f"max_n_samples: {max_n_samples}")
print(f"n_samples_step: {n_samples_step}")
print(f"min_n_qubits: {min_n_qubits}")
print(f"max_n_qubits: {max_n_qubits}")
print(f"n_runs: {n_runs}")

try:
    with open("results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}
    pass

if len(results) == 0:
    try:
        for n_qubits in range(min_n_qubits, max_n_qubits + 1):
            results[n_qubits] = {}
            for n_samples in range(min_n_samples, max_n_samples + 1, n_samples_step):
                results[n_qubits][n_samples] = {}
                rng_s = np.random.default_rng(seed)
                rng_p = np.random.default_rng(seed)
                for run in range(n_runs):
                    model = Model(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        circuit_type="Circuit_19",
                        random_seed=seed,
                    )
                    model.initialize_params(rng=rng_s, repeat=n_samples)

                    start = time_measure()
                    model(execution_type="density")
                    t_single = time_measure() - start

                    model = Model(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        circuit_type="Circuit_19",
                        mp_threshold=1000,
                        random_seed=seed,
                    )

                    model.initialize_params(rng=rng_p, repeat=n_samples)

                    start = time_measure()
                    model(execution_type="density")
                    t_parallel = time_measure() - start

                    print(
                        f"{run} | {n_qubits} qubits | {n_samples} samples: {t_single / t_parallel}"
                    )

                    results[n_qubits][n_samples][run] = t_single / t_parallel
    except KeyboardInterrupt:
        pass

    with open("results.json", "w") as f:
        json.dump(results, f)


for n_qubits in results.keys():
    y_mean = []
    y_max = []
    y_min = []
    for n_samples in results[n_qubits].keys():
        samples = list(results[n_qubits][n_samples].values())
        y_mean.append(np.mean(samples))

    std = np.std(y_mean)
    for y_mean_i in y_mean:
        y_max.append(y_mean_i + std)
        y_min.append(y_mean_i - std)

    plt.plot(
        list(results[n_qubits].keys()),
        y_mean,
        label=f"{n_qubits} qubits",
    )

    plt.fill_between(
        list(results[n_qubits].keys()),
        y_min,
        y_max,
        alpha=0.2,
    )

plt.xlabel("Number of samples")
plt.ylabel("Speedup")
plt.legend()
plt.savefig("figures/speedup_light.svg", dpi=100, transparent=True)
