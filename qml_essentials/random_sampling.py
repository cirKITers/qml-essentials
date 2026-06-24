import jax
import jax.numpy as jnp
from typing import Optional
from qml_essentials.operations import _cdtype

_DEFAULT_SEED = 1000


def _ginibre(key: jax.random.PRNGKey, rows: int, cols: int) -> jnp.ndarray:
    r"""Draw a complex Ginibre matrix :math:`G \in \mathbb{C}^{rows \times cols}`.

    Each entry is :math:`G_{jk} = a_{jk} + i\,b_{jk}` with
    :math:`a_{jk}, b_{jk} \sim \mathcal{N}(0, 1)` i.i.d. standard normal (the
    Ginibre ensemble). The result is cast to the active complex dtype via
    :func:`qml_essentials.operations._cdtype`.

    Args:
        key (jax.random.PRNGKey): JAX random key, split internally into a real
            and an imaginary part.
        rows (int): Number of rows :math:`d`.
        cols (int): Number of columns :math:`K`.

    Returns:
        jnp.ndarray: Complex Ginibre matrix of shape ``(rows, cols)``.
    """
    key_re, key_im = jax.random.split(key)
    real = jax.random.normal(key_re, shape=(rows, cols))
    imag = jax.random.normal(key_im, shape=(rows, cols))
    return (real + 1j * imag).astype(_cdtype())


def _haar_unitary(key: jax.random.PRNGKey, dim: int) -> jnp.ndarray:
    r"""Draw a Haar-random unitary :math:`U \in U(d)`.

    Implements the QR construction of Mezzadri (*How to generate random
    matrices from the classical compact groups*, arXiv:math-ph/0609050): draw a
    Ginibre matrix :math:`Z`, take its QR decomposition :math:`Z = QR`, and
    remove the phase ambiguity via
    :math:`U = Q\,\Lambda`, :math:`\Lambda = \mathrm{diag}(R_{ii}/|R_{ii}|)`,
    so that :math:`U` is Haar-distributed.

    Args:
        key (jax.random.PRNGKey): JAX random key for the Ginibre draw.
        dim (int): Dimension :math:`d` of the unitary.

    Returns:
        jnp.ndarray: Haar-random unitary of shape ``(dim, dim)``.
    """
    z = _ginibre(key, dim, dim)
    q, r = jnp.linalg.qr(z)
    diag_r = jnp.diagonal(r)
    # Guard against a zero pivot (|0| -> 1 keeps the phase at 1).
    phases = diag_r / jnp.where(jnp.abs(diag_r) > 0, jnp.abs(diag_r), 1.0)
    return q * phases[None, :]


def _normalize_density(rho: jnp.ndarray) -> jnp.ndarray:
    r"""Hermitize and trace-normalize a single density matrix.

    Applies :math:`\rho \leftarrow (\rho + \rho^\dagger)/2` to remove numerical
    anti-Hermitian noise, then divides by :math:`\mathrm{Tr}\,\rho` so that
    :math:`\mathrm{Tr}\,\rho = 1`.

    Args:
        rho (jnp.ndarray): Unnormalized matrix of shape ``(d, d)``.

    Returns:
        jnp.ndarray: Hermitian, unit-trace density matrix of shape ``(d, d)``.
    """
    rho = (rho + jnp.conj(rho).T) / 2.0
    return rho / jnp.trace(rho)


class DensityMatrix:
    r"""Random density-matrix samplers.

    Samplers for the standard random-density-matrix ensembles, built from the
    Ginibre construction
    :math:`\rho = G G^\dagger / \mathrm{Tr}(G G^\dagger)` and/or a Haar-random
    unitary. Every sampler returns a batched array of shape
    ``(n_samples, 2**n_qubits, 2**n_qubits)``, consistent with
    :func:`qml_essentials.entanglement.sample_random_separable_states`.

    The module is structured so that ``StateVector``, ``Hermitian`` and
    ``Unitary`` sampler classes can be added later, reusing the module-level
    primitives :func:`_ginibre` and :func:`_haar_unitary`.
    """

    @classmethod
    def induced(
        cls,
        n_qubits: int,
        n_samples: int = 1,
        rank: Optional[int] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        r"""Sample from the induced measure via the Ginibre construction.

        Draws :math:`\rho = G G^\dagger / \mathrm{Tr}(G G^\dagger)` with
        :math:`G` a :math:`d \times K` complex Ginibre matrix,
        :math:`d = 2^{n\_qubits}` and :math:`K = \mathrm{rank}`. This is the
        induced measure of Zyczkowski & Sommers (*Induced measures in the space
        of mixed quantum states*, arXiv:quant-ph/0012101). The sampled state
        has rank :math:`\min(d, K)` almost surely: :math:`K = 1` yields (Haar)
        pure states and :math:`K = d` recovers the Hilbert-Schmidt measure. The
        mean purity is :math:`\mathbb{E}[\mathrm{Tr}\,\rho^2] = (d + K)/(dK + 1)`.

        Args:
            n_qubits (int): Number of qubits; :math:`d = 2^{n\_qubits}`.
            n_samples (int): Number of density matrices to draw. Defaults to 1.
            rank (Optional[int]): Number of Ginibre columns :math:`K`. When
                ``None`` (default) it is set to :math:`d`, recovering the
                Hilbert-Schmidt measure. Must satisfy ``rank >= 1``.
            random_key (Optional[jax.random.PRNGKey]): JAX random key. When
                ``None``, falls back to ``jax.random.key(1000)`` (matching the
                default ``random_seed`` of :class:`~qml_essentials.model.Model`).

        Returns:
            jnp.ndarray: Density matrices of shape
            ``(n_samples, 2**n_qubits, 2**n_qubits)``.

        Raises:
            ValueError: If ``rank`` is given and ``rank < 1``.
        """
        d = 2**n_qubits
        if rank is None:
            rank = d
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}.")
        if random_key is None:
            random_key = jax.random.key(_DEFAULT_SEED)

        def _sample(key: jax.random.PRNGKey) -> jnp.ndarray:
            g = _ginibre(key, d, rank)
            rho = g @ jnp.conj(g).T
            return _normalize_density(rho)

        keys = jax.random.split(random_key, n_samples)
        return jax.vmap(_sample)(keys)

    @classmethod
    def hilbert_schmidt(
        cls,
        n_qubits: int,
        n_samples: int = 1,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        r"""Sample from the Hilbert-Schmidt measure.

        Special case of the induced measure with :math:`K = d`:
        :math:`\rho = G G^\dagger / \mathrm{Tr}(G G^\dagger)` with :math:`G` a
        square :math:`d \times d` complex Ginibre matrix. The Hilbert-Schmidt
        measure is the flat measure induced by the Hilbert-Schmidt metric
        (Zyczkowski & Sommers, arXiv:quant-ph/0012101). Delegates to
        :meth:`induced` with ``rank = d``.

        Args:
            n_qubits (int): Number of qubits; :math:`d = 2^{n\_qubits}`.
            n_samples (int): Number of density matrices to draw. Defaults to 1.
            random_key (Optional[jax.random.PRNGKey]): JAX random key. When
                ``None``, falls back to ``jax.random.key(1000)``.

        Returns:
            jnp.ndarray: Density matrices of shape
            ``(n_samples, 2**n_qubits, 2**n_qubits)``.
        """
        return cls.induced(
            n_qubits=n_qubits,
            n_samples=n_samples,
            rank=2**n_qubits,
            random_key=random_key,
        )

    @classmethod
    def bures(
        cls,
        n_qubits: int,
        n_samples: int = 1,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        r"""Sample from the Bures measure.

        Uses the construction of Osipov, Sommers & Zyczkowski (*Random Bures
        mixed states and the distribution of their purity*, arXiv:1004.1655):

        .. math::
            \rho = \frac{(\mathbb{1} + U)\, G G^\dagger\, (\mathbb{1} + U)^\dagger}
                        {\mathrm{Tr}\!\left[(\mathbb{1} + U)\, G G^\dagger\,
                        (\mathbb{1} + U)^\dagger\right]},

        where :math:`G` is a square :math:`d \times d` complex Ginibre matrix
        and :math:`U` is an independently drawn Haar-random unitary. The Bures
        measure is induced by the Bures (statistical-distance) metric.

        Args:
            n_qubits (int): Number of qubits; :math:`d = 2^{n\_qubits}`.
            n_samples (int): Number of density matrices to draw. Defaults to 1.
            random_key (Optional[jax.random.PRNGKey]): JAX random key. When
                ``None``, falls back to ``jax.random.key(1000)``.

        Returns:
            jnp.ndarray: Density matrices of shape
            ``(n_samples, 2**n_qubits, 2**n_qubits)``.
        """
        d = 2**n_qubits
        if random_key is None:
            random_key = jax.random.key(_DEFAULT_SEED)

        eye = jnp.eye(d, dtype=_cdtype())

        def _sample(key: jax.random.PRNGKey) -> jnp.ndarray:
            key_g, key_u = jax.random.split(key)
            g = _ginibre(key_g, d, d)
            u = _haar_unitary(key_u, d)
            a = eye + u
            rho = a @ (g @ jnp.conj(g).T) @ jnp.conj(a).T
            return _normalize_density(rho)

        keys = jax.random.split(random_key, n_samples)
        return jax.vmap(_sample)(keys)

    @classmethod
    def eigen(
        cls,
        n_qubits: int,
        n_samples: int = 1,
        alpha: float = 1.0,
        eigenvalues: Optional[jnp.ndarray] = None,
        random_key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        r"""Sample density matrices with a prescribed or Dirichlet spectrum.

        Builds :math:`\rho = U\,\mathrm{diag}(\lambda)\,U^\dagger` with :math:`U`
        Haar-random. The eigenvalue vector :math:`\lambda` is either supplied
        via ``eigenvalues`` (then identical for every sample), or drawn per
        sample from a symmetric Dirichlet distribution
        :math:`\lambda \sim \mathrm{Dir}(\alpha \mathbf{1}_d)`. The default
        :math:`\alpha = 1` gives the uniform (flat) distribution on the
        probability simplex.

        Note that this ensemble is, by construction, different from the
        Hilbert-Schmidt and induced measures: it lacks their eigenvalue
        repulsion (the squared Vandermonde factor).

        Args:
            n_qubits (int): Number of qubits; :math:`d = 2^{n\_qubits}`.
            n_samples (int): Number of density matrices to draw. Defaults to 1.
            alpha (float): Symmetric Dirichlet concentration parameter, used
                only when ``eigenvalues`` is ``None``. Defaults to ``1.0``.
            eigenvalues (Optional[jnp.ndarray]): Optional fixed spectrum of
                length :math:`d`. Entries must be nonnegative and sum to 1. When
                given, the same spectrum is used for every sample and only
                :math:`U` is randomized.
            random_key (Optional[jax.random.PRNGKey]): JAX random key. When
                ``None``, falls back to ``jax.random.key(1000)``.

        Returns:
            jnp.ndarray: Density matrices of shape
            ``(n_samples, 2**n_qubits, 2**n_qubits)``.

        Raises:
            ValueError: If ``eigenvalues`` is supplied and is not a
                length-:math:`d` vector of nonnegative entries summing to 1.
        """
        d = 2**n_qubits
        if random_key is None:
            random_key = jax.random.key(_DEFAULT_SEED)

        fixed_eigs = None
        if eigenvalues is not None:
            fixed_eigs = cls._validate_eigenvalues(eigenvalues, d)

        def _sample(key: jax.random.PRNGKey) -> jnp.ndarray:
            key_u, key_lam = jax.random.split(key)
            u = _haar_unitary(key_u, d)
            if fixed_eigs is None:
                lam = jax.random.dirichlet(key_lam, alpha * jnp.ones(d))
            else:
                lam = fixed_eigs
            rho = (u * lam[None, :].astype(_cdtype())) @ jnp.conj(u).T
            return _normalize_density(rho)

        keys = jax.random.split(random_key, n_samples)
        return jax.vmap(_sample)(keys)

    @classmethod
    def _validate_eigenvalues(cls, eigenvalues: jnp.ndarray, dim: int) -> jnp.ndarray:
        r"""Validate a user-supplied eigenvalue (spectrum) vector.

        Checks that the vector has length :math:`d`, is real and nonnegative,
        and sums to 1 (within tolerance).

        Args:
            eigenvalues (jnp.ndarray): Candidate spectrum.
            dim (int): Expected length :math:`d = 2^{n\_qubits}`.

        Returns:
            jnp.ndarray: Validated, real eigenvalue vector of shape ``(dim,)``.

        Raises:
            ValueError: If the shape, nonnegativity, or unit-sum constraint is
                violated.
        """
        eigs = jnp.asarray(eigenvalues)
        if eigs.shape != (dim,):
            raise ValueError(f"eigenvalues must have shape ({dim},), got {eigs.shape}.")
        eigs = jnp.real(eigs)
        if bool(jnp.any(eigs < -1e-10)):
            raise ValueError("eigenvalues must be nonnegative.")
        if not bool(jnp.isclose(jnp.sum(eigs), 1.0, atol=1e-8)):
            raise ValueError(f"eigenvalues must sum to 1, got {float(jnp.sum(eigs))}.")
        return eigs
