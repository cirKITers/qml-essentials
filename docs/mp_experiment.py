import time
import pennylane.numpy as np
import json
from qml_essentials.model import Model
import matplotlib.pyplot as plt

seed = 1000
min_n_samples = 500
max_n_samples = 10000
n_samples_step = 500
n_qubits = 4
min_mp_threshold = 500
max_mp_threshold = 5000
mp_threshold_step = 500
n_layers = 1
n_runs = 8

time_measure = time.time


try:
    with open("mp_results.json", "r") as f:
        results = json.load(f)
    print("Found and loaded mp_results.json")
except FileNotFoundError:
    results = {}
    print("Configuration:")
    print(f"n_layers: {n_layers}")
    print(f"min_n_samples: {min_n_samples}")
    print(f"max_n_samples: {max_n_samples}")
    print(f"n_qubits: {n_qubits}")
    print(f"min_mp_threshold: {min_mp_threshold}")
    print(f"max_mp_threshold: {max_mp_threshold}")
    print(f"n_samples_step: {n_samples_step}")
    print(f"n_runs: {n_runs}")

    pass

if len(results) == 0:
    try:
        for mp_threshold in range(
            min_mp_threshold, max_mp_threshold + 1, mp_threshold_step
        ):
            results[mp_threshold] = {}
            for n_samples in range(min_n_samples, max_n_samples + 1, n_samples_step):
                results[mp_threshold][n_samples] = {}
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
                        mp_threshold=mp_threshold,
                        random_seed=seed,
                    )

                    model.initialize_params(rng=rng_p, repeat=n_samples)

                    start = time_measure()
                    model(execution_type="density")
                    t_parallel = time_measure() - start

                    print(
                        f"{run} | {mp_threshold}/{max_mp_threshold} mp | {n_samples}/{max_n_samples} samples: {t_single / t_parallel:.2f}"
                    )

                    results[mp_threshold][n_samples][run] = t_single / t_parallel
    except KeyboardInterrupt:
        pass

    with open("mp_results.json", "w") as f:
        json.dump(results, f)


for mp_threshold in results.keys():
    y_mean = []
    y_max = []
    y_min = []
    for n_samples in results[mp_threshold].keys():
        samples = list(results[mp_threshold][n_samples].values())
        y_mean.append(np.mean(samples))

    std = np.std(y_mean)
    for y_mean_i in y_mean:
        y_max.append(y_mean_i + std)
        y_min.append(y_mean_i - std)

    plt.plot(
        list(results[mp_threshold].keys()),
        y_mean,
        label=f"{int(mp_threshold)} mp",
    )

    plt.fill_between(
        list(results[mp_threshold].keys()),
        y_min,
        y_max,
        alpha=0.2,
    )

ax = plt.gca()
ax.tick_params("x", rotation=45)
plt.xlabel("Number of samples")
plt.ylabel("Speedup")
plt.legend(
    loc="upper center",
    ncol=5,
    bbox_to_anchor=(0.48, 1.18),
    fancybox=True,
    framealpha=0.0,
)
plt.tight_layout()
plt.savefig("figures/mp_result_light.png", dpi=100, transparent=True)
