import jax
import jax.numpy as jnp
import pennylane as qml
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker
import csv

from qml_essentials.yaqsi import (
    QuantumScript,
)
from qml_essentials.operations import (
    H,
    CRX,
    PauliZ,
)

import logging

logger = logging.getLogger(__name__)
rng = jax.random.PRNGKey(1000)

identifier = datetime.now().strftime("%Y%m%d%H%M%S")
print(f"Identifier: {identifier}")

LOAD_LATEST = True  # Set to True to skip computation and load the latest CSV instead

qubit_sizes = list(range(3, 14))
modes = ["probs", "expval", "state", "density"]
n_iters = 100
batch_size = 100


def var_ghz_benchmark(mode, q) -> None:
    """Benchmark comparison with pennylane framework (parametric, batched).

    Simulates a realistic training loop: parameters change every iteration,
    batch of 100 samples is vmapped, and the same QuantumScript instance is
    reused across iterations to test JIT compilation caching.
    """

    global rng
    n_qubits = q
    rng, subkey = jax.random.split(rng)

    print(f"Running Yaqsi benchmark (mode: {mode}, {q} qubits)")
    # Pre-generate different parameters for each iteration to simulate
    # a training loop where params change every step.
    all_phis = jax.random.uniform(
        subkey, shape=(n_iters, batch_size), minval=-jnp.pi, maxval=jnp.pi
    )

    # --- Yaqsi ---
    def yaqsi_circuit(phi):
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CRX(phi, wires=[i, (i + 1) % n_qubits])

    # Reuse the same QuantumScript to benefit from JIT compilation caching
    script = QuantumScript(f=yaqsi_circuit)

    # Warmup (triggers first compilation)
    obs = [PauliZ(wires=i, record=False) for i in range(n_qubits)]
    _ = script.execute(type=mode, obs=obs, args=(all_phis[0],), in_axes=(0,))

    start = time.time()
    for i in range(n_iters):
        res_ys = script.execute(type=mode, obs=obs, args=(all_phis[i],), in_axes=(0,))
    t_ys = (time.time() - start) / n_iters
    print(
        f"Yaqsi {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_ys*1000:.2f} ms"
    )

    # --- PennyLane ---
    dev = qml.device("default.qubit", wires=n_qubits)

    pl_return_map = {
        "density": lambda: qml.density_matrix(wires=range(n_qubits)),
        "state": lambda: qml.state(),
        "probs": lambda: qml.probs(wires=range(n_qubits)),
        "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
    }

    @qml.qnode(dev)
    def pl_circuit(phi):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.CRX(phi, wires=[i, (i + 1) % n_qubits])
        return pl_return_map[mode]()

    # Warmup
    _ = pl_circuit(all_phis[0])

    start = time.time()
    for i in range(n_iters):
        res_pl = pl_circuit(all_phis[i])
    t_pl = (time.time() - start) / n_iters
    print(
        f"PennyLane {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_pl*1000:.2f} ms"
    )
    print(f"Ratio pl/yaqsi: {t_pl/t_ys:.2f}x")
    res_pl_arr = jnp.array(res_pl)
    # PennyLane expval returns (n_obs, batch) while yaqsi returns (batch, n_obs)
    if res_pl_arr.shape != res_ys.shape:
        res_pl_arr = res_pl_arr.T

    assert jnp.allclose(res_ys, res_pl_arr, atol=1e-10), "Results do not match"
    print(f"Results match")

    return [t_ys, t_pl]


# results[mode] = {"ys": [...], "pl": [...]} indexed by qubit_sizes
results = {mode: {"ys": [], "pl": []} for mode in modes}

if LOAD_LATEST:
    # Find the most recent benchmarks CSV in the current directory
    import glob

    csv_files = sorted(glob.glob("benchmarks-*.csv"))
    if not csv_files:
        raise FileNotFoundError("No benchmarks-*.csv files found to load.")
    latest_csv = csv_files[-1]
    identifier = latest_csv[len("benchmarks-") : -len(".csv")]
    print(f"Loading latest results from {latest_csv} (identifier: {identifier})")

    with open(latest_csv, newline="") as f:
        reader = csv.DictReader(f)
        # Reconstruct qubit_sizes from the file to keep ordering consistent
        qubit_sizes_seen = []
        rows = list(reader)
        for row in rows:
            q = int(row["n_qubits"])
            if q not in qubit_sizes_seen:
                qubit_sizes_seen.append(q)
        qubit_sizes = qubit_sizes_seen

        # Re-initialise results with the actual modes present in the file
        for row in rows:
            mode = row["mode"]
            if mode not in results:
                results[mode] = {"ys": [], "pl": []}

        for mode in results:
            mode_rows = [r for r in rows if r["mode"] == mode]
            results[mode]["ys"] = [float(r["t_ys_ms"]) for r in mode_rows]
            results[mode]["pl"] = [float(r["t_pl_ms"]) for r in mode_rows]
else:
    for q in qubit_sizes:
        for mode in modes:
            t_ys, t_pl = var_ghz_benchmark(mode, q)
            results[mode]["ys"].append(t_ys * 1000)  # convert to ms
            results[mode]["pl"].append(t_pl * 1000)

    # --- CSV export ---
    print(f"Exporting results to benchmarks-{identifier}.csv")
    with open(f"benchmarks-{identifier}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_qubits", "mode", "t_ys_ms", "t_pl_ms", "ratio_pl_ys"])
        for q_idx, q in enumerate(qubit_sizes):
            for mode in modes:
                t_ys_ms = results[mode]["ys"][q_idx]
                t_pl_ms = results[mode]["pl"][q_idx]
                writer.writerow(
                    [
                        q,
                        mode,
                        t_ys_ms,
                        t_pl_ms,
                        t_pl_ms / t_ys_ms,
                    ]
                )

# --- Matplotlib figure ---
print(f"Plotting results to benchmarks-{identifier}.png")


mode_colors = {
    "probs": "#1f77b4",
    "expval": "#ff7f0e",
    "state": "#2ca02c",
    "density": "#d62728",
}

fig, ax = plt.subplots(figsize=(9, 5))

for mode in modes:
    color = mode_colors[mode]
    ratio = [pl / ys for ys, pl in zip(results[mode]["ys"], results[mode]["pl"])]
    ax.plot(
        qubit_sizes,
        ratio,
        color=color,
        linestyle="-",
        marker="o",
        linewidth=2,
    )

# Reference line at ratio = 1 (equal performance)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=2, label="_nolegend_")

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Time ratio  PennyLane / Yaqsi")
ax.set_title(
    f"Yaqsi vs PennyLane - Rel., Parametric, Avg. over {n_iters} Iters, {batch_size} Batches"
)
# ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(bottom=1.0)
ax.set_xticks(qubit_sizes)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(True, linestyle=":", alpha=0.6)

# --- Grouped legend ---
# Group 1: one entry per mode (color patches)
mode_handles = [
    mlines.Line2D(
        [], [], color=mode_colors[m], linestyle="-", marker="o", linewidth=2, label=m
    )
    for m in modes
]

ax.legend(
    handles=mode_handles,
    title="Simulation Mode",
    loc="lower left",
    fontsize=9,
    title_fontsize=10,
)

plt.tight_layout()
plt.show()
plt.savefig(f"benchmarks-{identifier}.png", dpi=150)
print("Figure saved to benchmarks.png")
