# Random Sampling

Our package provides samplers for random density matrices drawn from the
standard ensembles on the space of mixed quantum states.
These are useful for benchmarking, statistical studies, and as reference
distributions.

All samplers live in the `DensityMatrix` class and return a batched array of
shape `(n_samples, 2**n_qubits, 2**n_qubits)`:

```python
import jax
from qml_essentials.random_sampling import DensityMatrix

rhos = DensityMatrix.hilbert_schmidt(
    n_qubits=2, n_samples=1000, random_key=jax.random.key(1000)
)
```

Here, `n_samples` is the number of density matrices to draw and `random_key` is an optional JAX random key. If not provided, it defaults to `jax.random.key(1000)`.

The module is structured so that it can later be extended with `StateVector`, `Hermitian`, and `Unitary` samplers, reusing the same low-level primitives.

## Hilbert-Schmidt

The Hilbert-Schmidt measure is the flat measure induced by the Hilbert-Schmidt metric. 
A sample is obtained from a square $d \times d$ complex Ginibre matrix $G$ (with $d = 2^{n_\text{qubits}}$) via

$$
\rho = \frac{G G^\dagger}{\mathrm{Tr}(G G^\dagger)}.
$$

The (complex) Ginibre matrix is a random matrix whose entries are independent Gaussian random variables, s.t. $\mathbb{E}\left[\left|G_{i j}\right|^{2}\right]=1$.

```python
rhos = DensityMatrix.hilbert_schmidt(
    n_qubits=2, n_samples=1000, random_key=jax.random.key(1000)
)
```

## Induced

The induced measure $\mu_{d,K}$ generalizes the Hilbert-Schmidt measure by letting the Ginibre matrix $G$ be rectangular, of shape $d \times K$, where $K$ is the `rank` parameter:

$$
\rho = \frac{G G^\dagger}{\mathrm{Tr}(G G^\dagger)}, \qquad G \in \mathbb{C}^{d \times K}.
$$

The sampled state has rank $\min(d, K)$ almost surely. $K = d$ recovers the Hilbert-Schmidt measure, while $K = 1$ yields (Haar-random) pure states. 
The mean purity is $\mathbb{E}[\mathrm{Tr}\,\rho^2] = (d + K)/(dK + 1)$.

```python
# Low-rank (more mixed) sampling with K = 2
rhos = DensityMatrix.induced(
    n_qubits=2, n_samples=1000, rank=2, random_key=jax.random.key(1000)
)
```

This measure is described in [Zyczkowski & Sommers (2000) - Induced measures in the space of mixed quantum states](https://doi.org/10.1088/0305-4470/34/35/335).

## Bures

The Bures measure is induced by the Bures (statistical-distance) metric and is a natural prior of minimal information.
A sample combines a Ginibre matrix $G$ with an independently drawn Haar-random unitary $U$:

$$
\rho = \frac{(\mathbb{1} + U)\, G G^\dagger\, (\mathbb{1} + U)^\dagger}
            {\mathrm{Tr}\!\left[(\mathbb{1} + U)\, G G^\dagger\, (\mathbb{1} + U)^\dagger\right]}.
$$

```python
rhos = DensityMatrix.bures(
    n_qubits=2, n_samples=1000, random_key=jax.random.key(1000)
)
```

The construction and its mean purity $\mathbb{E}[\mathrm{Tr}\,\rho^2] = (5d^2 + 1)/(2d(d^2 + 2))$ are given in [Osipov, Sommers & Zyczkowski  (2009) - Random Bures mixed states and the distribution of their purity](https://doi.org/10.1088/1751-8113/43/5/055302) and [Sommers & Zyczkowski (2003) - Bures volume of the set of mixed quantum states](https://doi.org/10.1088/0305-4470/36/39/308).

## Eigenvalue Sampling

This method builds a density matrix from a Haar-random eigenbasis $U$ and a chosen spectrum $\lambda$:

$$
\rho = U\,\mathrm{diag}(\lambda)\,U^\dagger.
$$

By default, the eigenvalues are drawn from a symmetric Dirichlet distribution $\lambda \sim \mathrm{Dir}(\alpha \mathbf{1}_d)$, where $\alpha = 1$ corresponds to the uniform (flat) distribution on the probability simplex:

```python
rhos = DensityMatrix.eigen(
    n_qubits=2, n_samples=1000, alpha=1.0, random_key=jax.random.key(1000)
)
```

Note that, by construction, this ensemble differs from the Hilbert-Schmidt and induced measures: it lacks their eigenvalue repulsion.

Alternatively, a fixed spectrum can be supplied via `eigenvalues` (a nonnegative vector of length $d$ summing to 1). The same spectrum is then used for every sample and only the eigenbasis $U$ is randomized:

```python
import jax.numpy as jnp

rhos = DensityMatrix.eigen(
    n_qubits=2,
    n_samples=1000,
    eigenvalues=jnp.array([0.5, 0.3, 0.15, 0.05]),
    random_key=jax.random.key(1000),
)
```
