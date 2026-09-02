# Algebra and States

Beyond the metrics computed from a `Model`, the `algebra` and `states` modules provide
dynamical Lie algebra (DLA) tooling and state-preparation utilities used for
trainability and barren-plateau analysis.

## DLA and g-purity

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

An ansatz that saturates $\mathfrak{su}(2^n)$ has a closure of $4^n - 1$ words, which quickly becomes intractable to enumerate.
Pass `max_dim` to stop the growth early; a result of that length means $\dim \mathfrak{g} \geq$ `max_dim` and the basis is partial, so it is only meaningful as a dimension bound:

```python
capped = lie_closure_paulis(gens, max_dim=dim_so2n(n))
assert len(capped) == dim_so2n(n)      # so(2n) is reached exactly
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


## Permutation-symmetric operators and input states

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
from jaqsi import Script
from qml_essentials.ansaetze import Ansaetze
import jaqsi

def circ():
    Ansaetze.Permutation_Equivariant.build(np.array([0.7, 1.1, 0.5]), n)

script = Script(circ, n_qubits=n)
zs = script.execute(type="expval", obs=[jaqsi.PauliZ(q) for q in range(n)], initial_state=dicke_state(n, 2))
```

