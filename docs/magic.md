# Magic

Magic, or nonstabilizerness, quantifies how far a quantum state is from the set of stabilizer states, i.e. the states reachable with Clifford gates alone.
It is the resource that, together with entanglement, separates classically simulable circuits from universal quantum computation.
Our package estimates the magic of a given model through the stabilizer Renyi entropy.

## Stabilizer Renyi Entropy

The default measure is the second-order stabilizer Renyi entropy $M_2$, introduced by [Leone, Oliviero and Hamma](https://doi.org/10.1103/PhysRevLett.128.050402) and extended in [Leone, Bittel](https://doi.org/10.1103/PhysRevA.110.L040403).
For a pure state it is defined as

$$
M_2(\lvert\psi\rangle) = -\log\!\left(\frac{1}{2^n}\sum_{P\in\mathcal{P}_n} \langle\psi\lvert P\rvert\psi\rangle^{4}\right),
$$

where the sum runs over all $4^n$ Pauli strings $\mathcal{P}_n$ on $n$ qubits.
The value is non-negative and equals zero if and only if the state is a stabilizer state.
Unlike most other magic measures, $M_2$ requires no minimization over stabilizer decompositions and is a smooth, differentiable function of the state.

```python
from qml_essentials.model import Model
from qml_essentials.magic import Magic

model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Hardware_Efficient",
        )

magic = Magic.stabilizer_renyi_entropy(
    model, n_samples=1000, random_key=jax.random.key(1000)
)
```

Here, `n_samples` is the number of parameter samples, drawn according to the default initialization strategy of the model, and `random_key` is an optional JAX random key for parameter initialization.
If not provided, the model's internal random key is used.
If you set `n_samples=None`, the currently stored parameters of the model are used to estimate the magic of a single state.
Setting `scale=True` multiplies `n_samples` by $2^n$.

As with the other measures, keyword arguments are passed through to the model call.

This measure is faithful only for pure states.
The states are obtained via `execution_type="state"`; passing `noise_params` (mixed states) produces values that are not a valid magic measure, and a warning is emitted.
Because the exact computation enumerates all $4^n$ Pauli expectation values, it is intended for small-to-moderate qubit counts.

## Non-Clifford count

As a cheap, complementary diagnostic, the number of non-Clifford gates in the circuit can be obtained via

```python
count = Magic.non_clifford_count(model)
```

This decomposes the circuit into Clifford and Pauli-rotation gates and counts the Pauli rotations.
It is only a coarse resource bound, not a faithful magic measure: it ignores the rotation angles and is therefore roughly constant across parameters, so two circuits with the same count can have very different magic.
Prefer `stabilizer_renyi_entropy` for an actual magic value.
