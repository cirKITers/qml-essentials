# YAQSI

This page aims to provide a brief overview of the YAQSI (yet another quantum simulator) embedded in our package.

The simulator aims to be fully abstracted by the `Model` class, so for most usecases, it should not be required to interact with the simulator directly.
However, some scenarios require building a custom circuits or require more granular control.

In the figure below, you can see how YAQSI provides the Foundation for the more standard interfaces `Model`, `Ansaetze` and `Gates`.
With the latter two being responsible of constructing quantum circuits and therefore interface direction with the `Operations` module of YAQSI, `Model` interfaces with the `Script` class, the main interface for circuit execution.

Generally, all operations are registered on a `Tape` when being created in the context of a `Script` (see examples below).
All matrix definitions (including Kraus channels for noisy simulation) are registered in the `Operations` module. 

![overview](figures/yaqsi_overview_light.png#center#only-light)
![overview](figures/yaqsi_overview_dark.png#center#only-dark)

While the standard gate execution is quite straight-forward, the pulse simulation requires a bit more care.
Here we split up `PulseGates` (abstracted by the `Gates` class) into `PulseParams` and `PulseEnvelope` to get more fine grained control over the underlying implementation.
As a single source of truth for both, there is the `PulseInformation` class, providing valid combination of these two characteristics.

![overview](figures/yaqsi_pulse_light.png#center#only-light)
![overview](figures/yaqsi_pulse_dark.png#center#only-dark)


The API of our simulator is very similar to what one might be used to know from pennylane.
For a basic circuit execution, we have to do two imports:

```python
import qml_essentials.yaqsi as ys
import qml_essentials.operations as op
```

Next, we can create a circuit and specify the observable:

```python
def circuit():
    op.H(wires=0)
    op.CX(wires=[0, 1])

obs = [op.PauliZ(wires=0), op.PauliZ(wires=1)]
```

Finally, creating a `Script` and excute it will give us the probabilities for this standard Bell-Circuit:

```python
yss = ys.Script(circuit)
yss.execute(type="probs", obs=obs)
```

Parameterization of circuits is straightforward; you just have to pass the args to the `execute` function:

```python
import jax.numpy as jnp

n_qubits = 1

def circuit(phi, theta, omega):
    op.Rot(phi, theta, omega, wires=0)
    op.Rot(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi, wires=0)

obs = [op.PauliZ(wires=i) for i in range(n_qubits)]
yss = ys.Script(circuit)
yss.execute(type="expval", obs=obs, args=(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi))
```

Training those circuits is a breeze as we entirely build upon JAX and can just use OPTAX for this purpose:

```python
import optax as otx

def cost_fct(params):
    phi, theta, omega = params
    return yss.execute(type="expval", obs=[op.PauliZ(0)], args=(phi, theta, omega))[0]

params = jax.numpy.array([0.1, 0.2, 0.3])
opt = otx.adam(0.01)
opt_state = opt.init(params)

for epoch in range(1, 101):
    grads = jax.grad(cost_fct)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = otx.apply_updates(params, updates)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Cost: {cost_fct(params):.4f}")
```

Beyond this, you can conveniently call `.dagger()` or `.power()` on operations, ... 

```python
def circuit():
        op.RX(0.5, wires=0)
        op.RX(0.5, wires=0).dagger()
        op.PauliX(wires=0).power(2)

obs = [op.PauliZ(0)]
yss = ys.Script(circuit)
res = yss.execute(type="expval", obs=obs)

print(res) # we expect to end up in |0⟩ again
```

or combine them with different noise channels:

```python
def circuit():
    op.H(wires=0)
    op.CX(wires=[0, 1])
    op.DepolarizingChannel(0.1, wires=0)
    op.DepolarizingChannel(0.1, wires=1)

yss = ys.Script(circuit)
rho = yss.execute(type="density")
purity = jnp.real(jnp.trace(rho @ rho))
print(purity) # Purity should be < 1 
```

As [Pulse Gates](pulses.md) are built entirely upon YAQSI operations, you can also use those in the circuit to perform pulse level simulation:

```python
from qml_essentials.gates import PulseGates

def circuit(w):
    PulseGates.RX(w, wires=0)

obs = [op.PauliZ(0)]
yss = ys.Script(circuit)
res = yss.execute(type="expval", obs=obs, args=(jnp.pi*0.5,))
print(res) # expect sth. around 0 (but not too close)
```

Mixing pulse level simulation with noisy simulations is possible as well:

```python
def circuit(w):
    PulseGates.RX(w, wires=0)
    PulseGates.RY(w, wires=0)
    PulseGates.CX(wires=[0, 1])
    op.DepolarizingChannel(0.1, wires=0)
    op.DepolarizingChannel(0.1, wires=1)

yss = ys.Script(circuit)
res = yss.execute(type="density", args=(jnp.pi*0.5,))
purity = jnp.real(jnp.trace(rho @ rho))
print(purity) # Purity should be < 1 
```

You can visualize the pulses schedules, i.e. the sequence in which the pulses are applied on each qubit in the circuit, using the `draw` method.
Here, shaded areas represent the pulse shape/envelope (e.g. "Gaussian") of the pulse and the vertical line represents the time at which the pulse is applied.
Note that all gates are automatically decomposed into basis gates (e.g. `H` is decomposed into `RZ` and `RY`).

```python
def circuit(w):
    PulseGates.RX(w, wires=0)
    PulseGates.CZ(wires=0)
    PulseGates.H(wires=1)
    PulseGates.H(wires=1)

yss = ys.Script(circuit)

fig, axes = yss.draw(figure="pulse", args=(jnp.pi*0.5,))
```

![pulse-schedule](figures/pulse_schedule_light.png#center#only-light)
![pulse-schedule](figures/pulse_schedule_dark.png#center#only-dark)

## Performance: pulse-level gradient throughput

The pulse-level pipeline is compiled with JAX/XLA, but the underlying
ODE integration is *sequential by construction*: each Adam step
performs a `diffrax.Dopri8` solve over the gate duration, and the
adaptive step controller cannot be parallelized across time.  That
means a single `value_and_grad(loss)(params)` call typically saturates
**one** CPU core — multi-threading helps only inside per-step matrix
products (which are tiny: 8×8 for three qubits) and across batched
restarts (`vmap`).

The performance can be tuned on following different levels:

1. **`PulseInformation.set_rwa(True)`** — opt-in rotating-wave
   approximation.  Drops the fast counter-rotating terms in the
   interaction-picture Hamiltonian.
   Default is `False` (exact integration).

2. **`Yaqsi.set_solver_defaults(solver=...)`** — opt-in commutator-free
   Magnus integrator on a fixed `lax.scan` grid.  
   No RWA, exact `H_I(t)`, but trades the adaptive Dopri8 step
   controller for a fixed grid of `magnus_steps` substeps that fuses
   into a single XLA program — eliminating per-step Python overhead
   and host↔device sync entirely.

   * `solver="dopri8"` (default): adaptive Dormand-Prince 8(7).
   * `solver="magnus2"`: midpoint Magnus, one `expm` per step.
     Second-order: error scales as `h^2`.
   * `solver="magnus4"`: Blanes-Moan CFM4:2, two `expm` per step.
     Fourth-order: error scales as `h^4` (≈16× drop per N doubling).
     Typically the best accuracy/cost trade-off for smooth oscillatory
     pulse drives — `magnus_steps=512` reaches ≲1e-7 error on a
     standard `RX(\\pi/2)` Drag pulse.

   Both Magnus integrators preserve unitarity to machine precision
   regardless of step size.  Choose `magnus_steps` so that
   `h = T/N` resolves the fastest oscillation in `H(t)`
   (a few steps per period of `\\omega_c + \\omega_q`).

   ```python
   from qml_essentials.yaqsi import Yaqsi
   Yaqsi.set_solver_defaults(solver="magnus4", magnus_steps=512)
   ```

3. **`PulseInformation.set_frame("drive")`** — algebraic rewrite of
   the (non-RWA) coefficients via the product-to-sum identity
   `cos(\\omega_c t) cos(\\omega_q t) = 1/2[cos(\\Delta t) + cos(\\Sigma t)]` with
   `\\Delta = \\omega_c - \\omega_q`, `\\Sigma = \\omega_c + \\omega_q`.  Mathematically identical to
   the default `"lab"` form (no information lost, no RWA applied).
   Primary use: combined with `magnus2`/`magnus4`, the explicit
   slow/fast decomposition is sometimes numerically better-conditioned
   when the drive is detuned (`|\\Delta| << |\\Sigma|`).  Switching the frame does
   not change the result of an adaptive solve.

4. **XLA / OMP thread settings**.  Even on a single ODE solve, XLA
   can parallelise some matmul-heavy reductions if you allow it to.
   Reasonable defaults for a workstation:

   ```bash
   export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true \
                     intra_op_parallelism_threads=$(nproc)"
   export OMP_NUM_THREADS=$(nproc)
   ```

   For `dim <= 16` (<= 4 qubits) the per-step matmuls are too small to
   benefit much from threads; the dominant gain comes from `vmap`
   parallelism.

### Diagnostic helper

`qml_essentials.qoc.profile_pulse_pipeline(gate, rwa=...)` builds a
minimal pulse `Script` for the requested gate, JIT-compiles a
forward + `value_and_grad` pass, and prints wall-clock timings for
both compilation and steady-state evaluation:

```python
from qml_essentials.qoc import profile_pulse_pipeline

profile_pulse_pipeline("RX", rwa=False)  # exact
profile_pulse_pipeline("RX", rwa=True)   # RWA, fast benchmark mode
```
