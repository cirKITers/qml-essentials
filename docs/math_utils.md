# Math Utils

A collection of model-agnostic math tools, operating on density matrices or states directly.

```python
from qml_essentials.math import quantum_fisher_information, fubini_study_metric, fidelity, trace_distance, phase_difference
```

## Quantum Fisher Information

The Quantum Fisher Information (QFI) is the metric tensor of the state manifold evaluated at a specific parameter point $\theta$.
Because it depends on the derivatives of the state with respect to the parameters, it is computed from the state as a **function** of the parameters rather than from a single state.
The Jacobian is obtained via forward-mode automatic differentiation, which yields the complex Jacobian directly for the real-valued circuit parameters.

For a pure, normalised state $\ket{\psi(\theta)}$ the QFI is the Fubini-Study metric (scaled by four):

\[F_{ij} = 4\,\mathrm{Re}\left[\braket{\partial_i\psi | \partial_j\psi} - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}\right]\]

For a mixed state $\rho(\theta) = \sum_k p_k \ket{k}\bra{k}$ the QFI is given through the symmetric logarithmic derivative:

\[F_{ij} = 2 \sum_{k, l\,:\,p_k + p_l > 0} \frac{\mathrm{Re}\left(\braket{k | \partial_i\rho | l}\braket{l | \partial_j\rho | k}\right)}{p_k + p_l}\]

Both cases are handled by the same function, which dispatches on the kind of state returned by the provided callable.
Set the model's `execution_type` to `"state"` to obtain the pure-state QFI, or to `"density"` (e.g. for noisy circuits) to obtain the mixed-state QFI:

```python
from qml_essentials.model import Model
from qml_essentials.math import quantum_fisher_information

model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
model.execution_type = "state"

qfi = quantum_fisher_information(lambda p: model(params=p), model.params)
```

The result is a real, symmetric $(P, P)$ matrix, where $P$ is the total number of parameters (the parameter axes are flattened).
The state returned by the callable is assumed to be normalised, which the underlying simulator guarantees.

Note that `model.params` is passed in its native (batched) shape; the callable closes over any
data `inputs`, e.g. `lambda p: model(params=p, inputs=x)`.

## Fubini-Study Metric

The Fubini-Study metric is the real part of the quantum geometric tensor on the manifold of
pure states.
It is the underlying geometric object of the pure-state QFI and is related to it by a factor of
four, $F_{ij} = 4\,g_{ij}$:

\[g_{ij} = \mathrm{Re}\left[\braket{\partial_i\psi | \partial_j\psi} - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}\right]\]

`fubini_study_metric(state_fn, params)` follows the same calling convention as
`quantum_fisher_information` but, since the metric is only defined for pure states, requires
`state_fn` to return a state vector (`execution_type = "state"`):

```python
from qml_essentials.math import fubini_study_metric

g = fubini_study_metric(lambda p: model(params=p), model.params)
```

## State comparison utilities

The remaining helpers compare two given quantum states.

### Fidelity

`fidelity(state0, state1)` computes the fidelity between two states, accepting either state
vectors or density matrices.
For pure states it evaluates $F(\ket{\psi}, \ket{\phi}) = \left|\braket{\psi | \phi}\right|^2$,
while for density matrices it uses the Uhlmann fidelity
$F(\rho, \sigma) = \left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2$.
Both single states and batches of shape $(B, \dots)$ are supported.

### Trace distance

`trace_distance(state0, state1)` returns the trace distance between two density matrices,

\[T(\rho, \sigma) = \frac{1}{2} \sum_i \left|\lambda_i\right|\]

where $\lambda_i$ are the eigenvalues of $\rho - \sigma$.

### Phase difference

`phase_difference(state0, state1)` returns the phase $\arg\braket{\psi | \phi}$ between two
state vectors.
A value of zero indicates the two states differ by at most a real global factor.

### Matrix logarithm

`logm_v(A)` computes the matrix logarithm of a single matrix of shape $(d, d)$ or of each
matrix in a batch of shape $(B, d, d)$, as used internally by the entropy-based measures.
