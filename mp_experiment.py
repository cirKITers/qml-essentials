import time
import numpy as np
from jax import random
import json
from qml_essentials.model import Model
import matplotlib.pyplot as plt
import argparse


time_measure = time.time


def main(
    seed,
    n_layers,
    min_n_samples,
    max_n_samples,
    n_samples_step,
    n_runs,
    n_qubits,
    execution_type,
):
    try:
        with open(f"mp_results_{execution_type}.json", "r") as f:
            results = json.load(f)
        print(f"Found and loaded mp_results_{execution_type}.json")
    except FileNotFoundError:
        results = {}
        print("Configuration:")
        print(f"n_layers: {n_layers}")
        print(f"min_n_samples: {min_n_samples}")
        print(f"max_n_samples: {max_n_samples}")
        print(f"n_qubits: {n_qubits}")
        print(f"n_samples_step: {n_samples_step}")
        print(f"n_runs: {n_runs}")

        pass

    if len(results) == 0:
        try:
            results = {}
            for n_samples in range(min_n_samples, max_n_samples + 1, n_samples_step):
                results[n_samples] = {}
                random_key_s = random.key(seed)
                random_key_p = random.key(seed)
                for run in range(n_runs):
                    model = Model(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        circuit_type="Circuit_19",
                        random_seed=seed,
                    )
                    model.initialize_params(random_key_s, repeat=n_samples)

                    start = time_measure()
                    model(execution_type=execution_type)
                    t_single = time_measure() - start

                    model = Model(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        circuit_type="Circuit_19",
                        random_seed=seed,
                        use_multithreading=True,
                    )

                    model.initialize_params(random_key_p, repeat=n_samples)
                    random_key_s, random_key_p = random.split(random_key_s)

                    start = time_measure()
                    model(execution_type=execution_type)
                    t_parallel = time_measure() - start

                    print(
                        f"{run} | "
                        f"{n_samples}/{max_n_samples} samples: "
                        f"{t_single / t_parallel:.2f}"
                    )

                    results[n_samples][run] = t_single / t_parallel
        except KeyboardInterrupt:
            pass

        with open(f"mp_results_{execution_type}.json", "w") as f:
            json.dump(results, f)

    y_mean = []
    y_max = []
    y_min = []
    for n_samples in results.keys():
        samples = list(results[n_samples].values())
        y_mean.append(np.mean(samples))

    std = np.std(y_mean)
    for y_mean_i in y_mean:
        y_max.append(y_mean_i + std)
        y_min.append(y_mean_i - std)

    plt.plot(list(results.keys()), y_mean, label="Multi-Threading")

    plt.fill_between(
        list(results.keys()),
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
    plt.savefig(
        f"figures/mp_result_{execution_type}_light.png", dpi=100, transparent=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="random seed",
    )
    parser.add_argument(
        "--execution_type",
        type=str,
        default="expval",
        choices=["density", "expval"],
        help="execution type",
    )
    parser.add_argument(
        "--min_n_samples",
        type=int,
        default=3000,  # 500, 3000
        help="minimal number of samples",
    )
    parser.add_argument(
        "--max_n_samples",
        type=int,
        default=60000,  # 10000, 60000
        help="maximal number of samples",
    )
    parser.add_argument(
        "--n_samples_step",
        type=int,
        default=3000,  # 500, 3000
        help="step size for the number of samples",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=4,
        help="number of qubits",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="number of layers",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=8,
        help="number of runs",
    )
    args = parser.parse_args()

    main(
        seed=args.seed,
        execution_type=args.execution_type,
        min_n_samples=args.min_n_samples,
        max_n_samples=args.max_n_samples,
        n_samples_step=args.n_samples_step,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_runs=args.n_runs,
    )
