# JAQSI

This page aims to provide a brief overview of the JAQSI (just another quantum simulator) embedded in our package.

The simulator aims to be fully abstracted by the `Model` class, so for most usecases, it should not be required to interact with the simulator directly.
However, some scenarios require building a custom circuits or require more granular control.

In the figure below, you can see how JAQSI provides the Foundation for the more standard interfaces `Model`, `Ansaetze` and `Gates`.
With the latter two being responsible of constructing quantum circuits and therefore interface direction with the `Operations` module of JAQSI, `Model` interfaces with the `Script` class, the main interface for circuit execution.

Generally, all operations are registered on a `Tape` when being created in the context of a `Script` (see examples below).
All matrix definitions (including Kraus channels for noisy simulation) are registered in the `Operations` module. 

![overview](figures/jaqsi_overview_light.png#center#only-light)
![overview](figures/jaqsi_overview_dark.png#center#only-dark)

While the standard gate execution is quite straight-forward, the pulse simulation requires a bit more care.
Here we split up `PulseGates` (abstracted by the `Gates` class) into `PulseParams` and `PulseEnvelope` to get more fine grained control over the underlying implementation.
As a single source of truth for both, there is the `PulseInformation` class, providing valid combination of these two characteristics.

![overview](figures/jaqsi_pulse_light.png#center#only-light)
![overview](figures/jaqsi_pulse_dark.png#center#only-dark)

## Architecture

Internally, the simulator is split into a handful of modules, each with a single responsibility.
Together they form a pipeline that turns a circuit function into a measurement result.

- `operations.py` : the foundation. Defines every quantum operation: gates (`H`, `RX`, `CX`, `Rot`, ...), observables (`PauliZ`, `Hermitian`), Kraus noise channels (`DepolarizingChannel`, `AmplitudeDamping`, ...) and the (parametrized) Hamiltonians used for pulse evolution. Each `Operation` carries its matrix definition and knows how to apply itself to a statevector or density matrix via cached `einsum` contractions.
- `tape.py` : the recording layer. Holds the thread-local `Tape` onto which operations register themselves as they are created. A `recording()` context manager collects the operations built inside a circuit function into an ordered list : nothing is executed yet.
- `script.py` : the orchestrator. The `Script` class is the main entry point (and what `Model` builds upon). It records the circuit, infers the number of qubits, decides between pure and density-matrix simulation, dispatches measurements, and takes care of JIT caching, automatic batching (`vmap` with memory-aware chunking) and circuit drawing.
- `simulation.py` : the compute engine. A set of pure, stateless functions that run a recorded tape: `simulate_pure` (statevector), `simulate_mixed` (density matrix) and the measurement kernels (`measure_state`, `measure_density`, `sample_shots`). Being pure JAX functions, they are fully differentiable and `jit`/`vmap`-compatible.
- `memory.py` : memory accounting. Pure helpers that estimate the peak memory of a batched run and, when it would not fit in available RAM, split the batch into chunks that do (`estimate_peak_bytes`, `compute_chunk_size`, `execute_chunked`). `Script` calls these to drive its memory-aware `vmap` chunking.
- `evolution.py` : Hamiltonian time-evolution. The `Evolution` class builds gates that evolve a (parametrized) Hamiltonian in time, either analytically (`exp(-i t H)` for a static `H`) or by solving the Schrödinger equation with an adaptive `diffrax` solver or a fixed-step Magnus integrator. This module backs the pulse-level simulation.
- `jaqsi.py` : the entry-point module. Exposes `Script` for circuit building, the `Hamiltonian` factory for time-evolution sources, and a few pulse/gate-independent quantum-info helpers (`partial_trace`, `marginalize_probs`, `build_parity_observable`). Time evolution is invoked as a method on the Hamiltonian object (`hamiltonian.evolve(...)`); the `Evolution` engine is re-exported here for solver configuration (`Evolution.set_solver_defaults`).
- `algebra.py` : a companion module for dynamical Lie algebra (DLA) and trainability analysis, layered on `operations.py` rather than part of the simulate-measure pipeline. It builds DLAs from generators (`lie_closure_paulis`, `lie_closure_matrices`), constructs the matchgate algebra $\mathfrak{so}(2n)$ (`matchgate_generators`, `matchgate_basis`, `dim_so2n`), computes the g-purity of a state against a DLA basis (`g_purity_from_basis`, `g_purity_matrix`), and provides the permutation-symmetric operators `symmetric_pauli_sum`, `sn_equivariant_generators` and `sn_equivariant_observable`.
- `states.py` : state-preparation utilities returning dense statevectors of shape $(2^n,)$ (qubit 0 leftmost): the Dicke state (`dicke_state`), Haar-random states (`haar_state`) and graph states (`graph_state_vector` with the edge-set constructors `matching_edges`, `path_edges`, `complete_edges`). The arrays feed `Script.execute(initial_state=...)` and the `g_purity_*` helpers directly.

A call to `Script.execute(...)` then runs four stages:

1. Record : the circuit function is executed once so that each operation registers itself on a fresh `Tape`.
2. Prepare : the qubit count is inferred and the presence of noise channels decides between statevector and density-matrix simulation.
3. Simulate : the operations are applied in order, each gate contracted into the state via `einsum`.
4. Measure : the resulting state is turned into the requested output (`state`, `probs`, `expval` or `density`) and optionally sampled into shots.

As the whole pipeline is built on JAX, any execution can be differentiated, JIT-compiled and vectorized.

## Usage

The API of our simulator is very similar to what one might be used to know from pennylane.

### Gate Level

For a basic circuit execution, we have to do two imports:

```python
import qml_essentials.jaqsi as js
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
jss = js.Script(circuit)
jss.execute(type="probs", obs=obs)
```

Parameterization of circuits is straightforward; you just have to pass the args to the `execute` function:

```python
import jax.numpy as jnp

n_qubits = 1

def circuit(phi, theta, omega):
    op.Rot(phi, theta, omega, wires=0)
    op.Rot(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi, wires=0)

obs = [op.PauliZ(wires=i) for i in range(n_qubits)]
jss = js.Script(circuit)
jss.execute(type="expval", obs=obs, args=(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi))
```

Training those circuits is a breeze as we entirely build upon JAX and can just use OPTAX for this purpose:

```python
import optax as otx

def cost_fct(params):
    phi, theta, omega = params
    return jss.execute(type="expval", obs=[op.PauliZ(0)], args=(phi, theta, omega))[0]

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
jss = js.Script(circuit)
res = jss.execute(type="expval", obs=obs)

print(res) # we expect to end up in |0⟩ again
```

or combine them with different noise channels:

```python
def circuit():
    op.H(wires=0)
    op.CX(wires=[0, 1])
    op.DepolarizingChannel(0.1, wires=0)
    op.DepolarizingChannel(0.1, wires=1)

jss = js.Script(circuit)
rho = jss.execute(type="density")
purity = jnp.real(jnp.trace(rho @ rho))
print(purity) # Purity should be < 1 
```

By default the simulation starts from the all-zero state $\lvert 0\dots0\rangle$.
To start from an arbitrary statevector instead, pass it via the `initial_state`
argument of `execute`:

```python
def circuit():
    op.RX(0.3, wires=0)

jss = js.Script(circuit)
plus = jnp.array([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)  # |+⟩
res = jss.execute(type="expval", obs=[op.PauliZ(0)], initial_state=plus)
```

Without `in_axes` the state must be a single statevector of shape `(2**n,)`. 
When batching with `in_axes`, `initial_state` may be a single 1D state broadcast across the batch, or a 2D array of shape `(B, 2**n)` that provides one state per sample.


### DLA and g-purity

Beyond circuit execution, the `algebra` module provides dynamical Lie algebra (DLA) helpers used for trainability and barren-plateau analysis.
The matchgate algebra $\mathfrak{so}(2n)$ is available both as a generating set and as an explicit Pauli-string basis, and the latter must match the Lie closure of the former:

```python
from qml_essentials.algebra import (
    matchgate_generators,
    matchgate_basis,
    dim_so2n,
    lie_closure_paulis,
    g_purity_from_basis,
)

n = 3
gens = matchgate_generators(n)         # {Z_k} u {X_k X_{k+1}}
basis = matchgate_basis(n)             # the n(2n-1) Pauli strings of so(2n)
assert len(basis) == dim_so2n(n)
assert {pw.to_pauli_string() for pw in lie_closure_paulis(gens)} == set(basis)
```

The g-purity $P_g = \sum_B \langle\psi\lvert B\rvert\psi\rangle^2$ of a statevector with respect to a DLA basis measures how much of the state lies in the algebra:

```python
import numpy as np

psi = np.zeros(2**n, dtype=complex)
psi[0] = 1.0                           # |0...0>
print(g_purity_from_basis(psi, basis)) # 3.0 (only the on-site Z_k contribute)
```

`g_purity_from_basis` takes a Pauli-word basis (strings or `PauliWord` objects).
For a Hilbert-Schmidt-orthonormal Hermitian matrix basis, for example the output of `lie_closure_matrices`, use `g_purity_matrix` instead.


### Permutation-symmetric operators and input states

The `symmetric_pauli_sum` constructor sums a Pauli over all subsets of a given size, e.g. $\sum_k X_k$ (`locality=1`) or $\sum_{j<k} X_j X_k$ (`locality=2`).
The $S_n$-equivariant generators $\{\sum_k X_k, \sum_k Y_k, \sum_{j<k} Z_j Z_k\}$ and observable $O = \tfrac{2}{n(n-1)} \sum_{j<k} X_j X_k$ build on it, and the generators feed the matrix DLA:

```python
import numpy as np
from qml_essentials.algebra import (
    sn_equivariant_generators,
    lie_closure_matrices,
    g_purity_matrix,
)
from qml_essentials.states import dicke_state, haar_state, graph_state_vector, path_edges

n = 4
basis = lie_closure_matrices(sn_equivariant_generators(n))   # HS-orthonormal DLA basis

for psi in (dicke_state(n, 2), haar_state(n, seed=0), graph_state_vector(n, path_edges(n))):
    print(g_purity_matrix(psi, basis))
```

The same statevectors can be evolved through a circuit by passing them as the initial state:

```python
from qml_essentials.jaqsi import Script
from qml_essentials.ansaetze import Ansaetze
from qml_essentials.operations import PauliZ

def circ():
    Ansaetze.Permutation_Equivariant.build(np.array([0.7, 1.1, 0.5]), n)

script = Script(circ, n_qubits=n)
zs = script.execute(type="expval", obs=[PauliZ(q) for q in range(n)], initial_state=dicke_state(n, 2))
```


### Pulse Level

This section is focussing on the Pulse-level related interface of the simulator.
If you want to work with the higher-level `Model` interface instead, head over to the [pulses](pulses.md) documentation.

As [Pulse Gates](pulses.md) are built entirely upon JAQSI operations, you can also use those in the circuit to perform pulse level simulation:

```python
from qml_essentials.gates import PulseGates

def circuit(w):
    PulseGates.RX(w, wires=0)

obs = [op.PauliZ(0)]
jss = js.Script(circuit)
res = jss.execute(type="expval", obs=obs, args=(jnp.pi*0.5,))
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

jss = js.Script(circuit)
res = jss.execute(type="density", args=(jnp.pi*0.5,))
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

jss = js.Script(circuit)

fig, axes = jss.draw(figure="pulse", args=(jnp.pi*0.5,))
```

![pulse-schedule](figures/pulse_schedule_light.png#center#only-light)
![pulse-schedule](figures/pulse_schedule_dark.png#center#only-dark)

Now let's get a level deeper into the pulse interface.
Under the hood what happens when you run a pulse gate, is that you evolve a Hermitian matrix in time.
To demonstrate this, we build a very simple circuit:

```python
def evol_circuit(t):
    time_evol = op.Hermitian(matrix=op.PauliZ._matrix, wires=0).evolve()
    time_evol(t=t, wires=0)
```

We can use this circuit directly in JAQSI by passing it to the `Script` class we've seen above:

```python
jss = js.Script(f=evol_circuit)
res = jss.execute(type="expval", obs=[PauliX(0)], args=(0.3,))
```

Here, we let the circuit evolve for `t=0.3` and measure the qubit in the `X` basis.
Obviously this isn't particluarly usefull, because it doesn't change the state of the qubit.
However, we can extend this circuit a little bit to start in the `|+⟩` state instead:

```python
def evol_circuit(t):
    H(wires=0)  # prepare |+⟩
    time_evol = op.Hermitian(matrix=op.PauliZ._matrix, wires=0).evolve()
    time_evol(t=t, wires=0)
```

Note, how we combine a "standard" gate here and combine it with a Hermitian evolution.
We can then measure:

```python
t = 0.3
jss = js.Script(f=evol_circuit)
res = jss.execute(type="expval", obs=[PauliX(0)], args=(t,))
```

which gives us exactly `jnp.cos(2 * t)`.

We've just seen an example for a static Hermitian evolution.
Naturally we can extend this to a parameterized Hermitian as well:

```python
def coeff(p, t):
    return p

def circuit(p,t):
    H(wires=0)  # prepare |+⟩
    ph = coeff * Hermitian(matrix=Z, wires=0, record=False)
    ph.evolve()([p], t)
```

Note here, that `coeff` is a callable.
While it seems a little bit strange to first use a callable and the parameterize it directly afterwards, this mechanism allows us to pre-compile the operation.

```python
jss = js.Script(f=circuit)
res = jss.execute(type="expval", obs=[PauliX(0)], args=(p,))
```

Naturally, we can now use this parameter in a training-scenario and leverage the performance advantage we got through the pre-compilation.