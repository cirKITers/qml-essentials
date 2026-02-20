import jax
import jax.numpy as jnp
import pennylane as qml
import time

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
rng = jax.random.PRNGKey(42)


def test_batch_benchmark(mode, q) -> None:
    """Benchmark comparison with pennylane framework (parametric, batched).

    Simulates a realistic training loop: parameters change every iteration,
    batch of 100 samples is vmapped, and the same QuantumScript instance is
    reused across iterations to test JIT compilation caching.
    """

    global rng
    n_qubits = q
    n_iters = 100
    batch_size = 1000
    rng, subkey = jax.random.split(rng)

    # Pre-generate different parameters for each iteration to simulate
    # a training loop where params change every step.
    all_phis = jax.random.uniform(
        subkey, shape=(n_iters, batch_size), minval=-jnp.pi, maxval=jnp.pi
    )

    obs = [PauliZ(wires=i, record=False) for i in range(n_qubits)]

    # --- Yaqsi ---
    def yaqsi_circuit(phi):
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CRX(phi, wires=[i, (i + 1) % n_qubits])

    # Reuse the same QuantumScript to benefit from JIT compilation caching
    script = QuantumScript(f=yaqsi_circuit)

    # Warmup (triggers first compilation)
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
    print(f"Ratio yaqsi/pl: {t_ys/t_pl:.1f}x")
    res_pl_arr = jnp.array(res_pl)
    # PennyLane expval returns (n_obs, batch) while yaqsi returns (batch, n_obs)
    if res_pl_arr.shape != res_ys.shape:
        res_pl_arr = res_pl_arr.T
    print(f"Results match: {jnp.allclose(res_ys, res_pl_arr, atol=1e-10)}")

    return [t_ys, t_pl]


import matplotlib.pyplot as plt
import csv

qubit_sizes = [4, 6, 8, 10, 12, 14]
modes = ["probs", "expval", "state", "density"]

# results[mode] = {"ys": [...], "pl": [...]} indexed by qubit_sizes
results = {mode: {"ys": [], "pl": []} for mode in modes}

for q in qubit_sizes:
    for mode in modes:
        t_ys, t_pl = test_batch_benchmark(mode, q)
        results[mode]["ys"].append(t_ys * 1000)  # convert to ms
        results[mode]["pl"].append(t_pl * 1000)

# --- CSV export ---
with open("benchmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_qubits", "mode", "t_ys_ms", "t_pl_ms", "ratio_ys_pl"])
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
                    t_ys_ms / t_pl_ms,
                ]
            )

# --- Matplotlib figure ---
import matplotlib.lines as mlines

mode_colors = {
    "probs": "#1f77b4",
    "expval": "#ff7f0e",
    "state": "#2ca02c",
    "density": "#d62728",
}

fig, ax = plt.subplots(figsize=(9, 5))

for mode in modes:
    color = mode_colors[mode]
    ratio = [ys / pl for ys, pl in zip(results[mode]["ys"], results[mode]["pl"])]
    ax.plot(
        qubit_sizes,
        ratio,
        color=color,
        linestyle="-",
        marker="o",
        linewidth=2,
    )

# Reference line at ratio = 1 (equal performance)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="_nolegend_")
ax.text(qubit_sizes[-1] + 0.1, 1.0, "parity", va="center", color="gray", fontsize=8)

ax.set_xlabel("Number of qubits")
ax.set_ylabel("Time ratio  Yaqsi / PennyLane")
ax.set_title("Benchmark: Yaqsi vs PennyLane – relative speed (batched, parametric)")
ax.set_xticks(qubit_sizes)
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
    title="Mode",
    loc="lower left",
    fontsize=8,
    title_fontsize=9,
)

plt.tight_layout()
plt.savefig("benchmarks.png", dpi=150)
plt.show()
print("Figure saved to benchmarks.png")
