import jax
import jax.numpy as jnp
import pennylane as qml
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker
import csv
import numpy as np

from qml_essentials.yaqsi import (
    Script,
)
from qml_essentials.operations import (
    H,
    CRX,
    PauliZ,
)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

identifier = datetime.now().strftime("%Y%m%d%H%M%S")
logger.info(f"Identifier: {identifier}")


rng = jax.random.PRNGKey(1000)

LOAD_LATEST = False  # Set to True to skip computation and load the latest CSV instead
WARMUP = True  # Does not produce meaningful results if False

qubit_sizes = list(range(14, 16))
modes = ["probs", "expval", "state", "density"]
n_iters = 100
batch_size = 10
precision = 1e-8


def var_ghz_benchmark(mode, q) -> None:
    """Benchmark comparison with pennylane framework (parametric, batched).

    Simulates a realistic training loop: parameters change every iteration,
    batch of 100 samples is vmapped, and the same Script instance is
    reused across iterations to test JIT compilation caching.
    """

    global rng
    n_qubits = q
    rng, subkey = jax.random.split(rng)

    logger.info(f"Running Yaqsi benchmark (mode: {mode}, {q} qubits)")
    # Pre-generate different parameters for each iteration to simulate
    # a training loop where params change every step.
    # Note that this is n_iters x batch_size, so every iteration is a batch
    # on its own!
    # We count the first iteration as warmup
    all_phis = jax.random.uniform(
        subkey, shape=(n_iters + 1, batch_size), minval=-jnp.pi, maxval=jnp.pi
    )

    # --- Yaqsi ---
    def yaqsi_circuit(phi):
        for i in range(n_qubits):
            H(wires=i)
        for i in range(n_qubits):
            CRX(phi, wires=[i, (i + 1) % n_qubits])

    script = Script(f=yaqsi_circuit)

    # do some warmup (allows pre-compilation)
    # Note that we use a phi, which will not be part of the benchmarking later
    if WARMUP:
        _ = script.execute(
            type=mode,
            obs=[PauliZ(wires=i, record=False) for i in range(n_qubits)],
            args=(all_phis[-1],),
            in_axes=(0,),
        )

    # do the actual benchmarking
    # Note that each execution uses a different phi vector, i.e. this is not
    # just repeating the same computation!
    ys_times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        res_ys = script.execute(
            type=mode,
            obs=[PauliZ(wires=i, record=False) for i in range(n_qubits)],
            args=(all_phis[i],),
            in_axes=(0,),
        )
        ys_times.append(time.perf_counter() - t0)
    t_ys = float(np.mean(ys_times))
    std_ys = float(np.std(ys_times))

    logger.info(
        f"Yaqsi {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_ys*1000:.2f} ± {std_ys*1000:.2f} ms"
    )

    # Now the same thing for pennylane
    # --- PennyLane ---
    dev = qml.device("default.qubit", wires=n_qubits)

    # Need to do a slight mapping for the different return types
    pl_return_map = {
        "density": lambda: qml.density_matrix(wires=range(n_qubits)),
        "state": lambda: qml.state(),
        "probs": lambda: qml.probs(wires=range(n_qubits)),
        "expval": lambda: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)],
    }

    @qml.qnode(dev, interface="jax")
    def pl_circuit(phi):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.CRX(phi, wires=[i, (i + 1) % n_qubits])
        return pl_return_map[mode]()

    if WARMUP:
        _ = pl_circuit(all_phis[-1])

    pl_times = []
    for i in range(n_iters):
        t0 = time.perf_counter()
        res_pl = pl_circuit(all_phis[i])
        pl_times.append(time.perf_counter() - t0)
    t_pl = float(np.mean(pl_times))
    std_pl = float(np.std(pl_times))

    logger.info(
        f"PennyLane {mode} ({n_qubits}q, batch={batch_size}, avg {n_iters}): "
        f"{t_pl*1000:.2f} ± {std_pl*1000:.2f} ms"
    )
    logger.info(f"Ratio pl/yaqsi: {t_pl/t_ys:.2f}x")

    res_pl_arr = jnp.array(res_pl)
    # PennyLane returns expval as (n_obs, batch) while Yaqsi returns
    # (batch, n_obs).  For other modes the shapes already agree.
    if mode == "expval":
        res_pl_arr = res_pl_arr.T

    if not jnp.allclose(res_ys, res_pl_arr, atol=precision):
        logger.error(
            f"Error occured at {q} qubits for mode {mode}:\
                     Results do not match; got {res_ys} and {res_pl_arr}\
                     Shape is {res_ys.shape} and {res_pl_arr.shape}"
        )
    else:
        logger.info("Results match")

    return [t_ys, t_pl, std_ys, std_pl]


# results[mode] = {"ys": [...], "pl": [...], "std_ys": [...], "std_pl": [...]} indexed by qubit_sizes
results = {mode: {"ys": [], "pl": [], "std_ys": [], "std_pl": []} for mode in modes}

if LOAD_LATEST:
    # Find the most recent benchmarks CSV in the current directory
    import glob

    csv_files = sorted(glob.glob("benchmarks-*.csv"))
    if not csv_files:
        raise FileNotFoundError("No benchmarks-*.csv files found to load.")
    latest_csv = csv_files[-1]
    identifier = latest_csv[len("benchmarks-") : -len(".csv")]
    logger.info(f"Loading latest results from {latest_csv} (identifier: {identifier})")

    with open(latest_csv, newline="") as f:
        reader = csv.DictReader(f)
        # Reconstruct qubit_sizes from the file to keep ordering consistent
        qubit_sizes_seen = []
        rows = list(reader)
        for row in rows:
            q_idx = int(row["n_qubits"])
            if q_idx not in qubit_sizes_seen:
                qubit_sizes_seen.append(q_idx)
        qubit_sizes = qubit_sizes_seen

        # Re-initialise results with the actual modes present in the file
        for row in rows:
            mode = row["mode"]
            if mode not in results:
                results[mode] = {"ys": [], "pl": [], "std_ys": [], "std_pl": []}

        for mode in results:
            mode_rows = [r for r in rows if r["mode"] == mode]
            results[mode]["ys"] = [float(r["t_ys_ms"]) for r in mode_rows]
            results[mode]["pl"] = [float(r["t_pl_ms"]) for r in mode_rows]
            results[mode]["std_ys"] = [float(r["std_ys_ms"]) for r in mode_rows]
            results[mode]["std_pl"] = [float(r["std_pl_ms"]) for r in mode_rows]
else:
    logger.info(f"Preparing header in benchmarks-{identifier}.csv")

    with open(f"benchmarks-{identifier}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_qubits",
                "mode",
                "t_ys_ms",
                "t_pl_ms",
                "std_ys_ms",
                "std_pl_ms",
                "ratio_pl_ys",
            ]
        )

    for q_idx in qubit_sizes:
        for mode in modes:
            t_ys, t_pl, std_ys, std_pl = var_ghz_benchmark(mode, q_idx)
            results[mode]["ys"].append(t_ys * 1000)  # convert to ms
            results[mode]["pl"].append(t_pl * 1000)
            results[mode]["std_ys"].append(std_ys * 1000)
            results[mode]["std_pl"].append(std_pl * 1000)

        # --- CSV export ---
        logger.info(f"Exporting results to benchmarks-{identifier}.csv")
        with open(f"benchmarks-{identifier}.csv", "a", newline="") as f:
            writer = csv.writer(f)
            for mode in modes:
                t_ys_ms = results[mode]["ys"][-1]
                t_pl_ms = results[mode]["pl"][-1]
                writer.writerow(
                    [
                        q_idx,
                        mode,
                        t_ys_ms,
                        t_pl_ms,
                        results[mode]["std_ys"][-1],
                        results[mode]["std_pl"][-1],
                        t_pl_ms / t_ys_ms,
                    ]
                )

# --- Matplotlib figure ---
logger.info(f"Plotting results to benchmarks-{identifier}.png")


mode_colors = {
    "probs": "#E69F00",
    "expval": "#ED665A",
    "state": "#009371",
    "density": "#002D4C",
}

fig, ax = plt.subplots(figsize=(9, 5))

for mode in modes:
    color = mode_colors[mode]
    ys = results[mode]["ys"]
    pl = results[mode]["pl"]
    std_ys = results[mode]["std_ys"]
    std_pl = results[mode]["std_pl"]

    ratio = [p / y for y, p in zip(ys, pl)]
    # Error propagation for ratio r = pl/ys:
    # σ_r = r * sqrt((σ_pl/pl)² + (σ_ys/ys)²)
    ratio_err = [
        r * ((sp / p) ** 2 + (sy / y) ** 2) ** 0.5
        for r, y, p, sy, sp in zip(ratio, ys, pl, std_ys, std_pl)
    ]

    ax.errorbar(
        qubit_sizes,
        ratio,
        yerr=ratio_err,
        color=color,
        linestyle="-",
        marker="o",
        linewidth=2,
        capsize=4,
        capthick=1.5,
        elinewidth=1.2,
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
logger.info("Figure saved to benchmarks.png")
