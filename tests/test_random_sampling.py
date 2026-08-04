import jax
import jax.numpy as jnp
import math
import pytest

from qml_essentials.random_sampling import DensityMatrix

jax.config.update("jax_enable_x64", True)

KEY = jax.random.key(1000)

# The four samplers share the common (n_qubits, n_samples, random_key) signature
# and can therefore be exercised uniformly by the validity/determinism/mean tests.
SAMPLERS = [
    DensityMatrix.hilbert_schmidt,
    DensityMatrix.induced,
    DensityMatrix.bures,
    DensityMatrix.eigen,
]
SAMPLER_IDS = ["hilbert_schmidt", "induced", "bures", "eigen"]


def _purity(rho: jnp.ndarray) -> jnp.ndarray:
    """Per-sample purity Tr(rho^2) for a batched density matrix."""
    return jnp.real(jnp.trace(rho @ rho, axis1=-2, axis2=-1))


@pytest.mark.unittest
@pytest.mark.parametrize("sampler", SAMPLERS, ids=SAMPLER_IDS)
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_sampler_validity(sampler, n_qubits) -> None:
    d = 2**n_qubits
    rho = sampler(n_qubits=n_qubits, n_samples=8, random_key=KEY)

    assert rho.shape == (8, d, d), "Wrong output shape"

    # Hermiticity
    assert jnp.allclose(rho, jnp.conj(jnp.swapaxes(rho, -1, -2)), atol=1e-10), (
        "Density matrices are not Hermitian"
    )

    # Unit trace
    traces = jnp.trace(rho, axis1=-2, axis2=-1)
    assert jnp.allclose(traces, 1.0 + 0j, atol=1e-10), "Trace is not 1"

    # Positive semi-definiteness
    eigvals = jnp.linalg.eigvalsh(rho).real
    assert float(jnp.min(eigvals)) >= -1e-10, "Density matrices are not PSD"


@pytest.mark.unittest
@pytest.mark.parametrize("sampler", SAMPLERS, ids=SAMPLER_IDS)
def test_determinism(sampler) -> None:
    a = sampler(n_qubits=2, n_samples=4, random_key=jax.random.key(1000))
    b = sampler(n_qubits=2, n_samples=4, random_key=jax.random.key(1000))
    assert jnp.array_equal(a, b), "Same key produced different samples"

    c = sampler(n_qubits=2, n_samples=4, random_key=jax.random.key(2000))
    assert not jnp.allclose(a, c), "Different keys produced identical samples"


@pytest.mark.unittest
@pytest.mark.parametrize("sampler", SAMPLERS, ids=SAMPLER_IDS)
@pytest.mark.parametrize("n_qubits, atol", [(1, 0.03), (2, 0.05)])
def test_mean_state_maximally_mixed(sampler, n_qubits, atol) -> None:
    # All four ensembles are unitarily invariant (eigen with the default
    # alpha=1 has Haar eigenvectors), so the ensemble mean is the maximally
    # mixed state I/d.
    d = 2**n_qubits
    rho = sampler(n_qubits=n_qubits, n_samples=4000, random_key=KEY)
    mean = rho.mean(axis=0)
    assert jnp.allclose(mean, jnp.eye(d) / d, atol=atol), (
        "Ensemble mean is not the maximally mixed state"
    )


@pytest.mark.unittest
@pytest.mark.parametrize("n_qubits, expected", [(1, 0.8), (2, 8.0 / 17.0)])
def test_purity_hilbert_schmidt(n_qubits, expected) -> None:
    # Hilbert-Schmidt mean purity E[Tr rho^2] = 2d/(d^2+1).
    rho = DensityMatrix.hilbert_schmidt(
        n_qubits=n_qubits, n_samples=4000, random_key=KEY
    )
    purity = float(_purity(rho).mean())
    assert math.isclose(purity, expected, abs_tol=0.02), (
        f"HS mean purity {purity} deviates from {expected}"
    )


@pytest.mark.unittest
@pytest.mark.parametrize(
    "n_qubits, rank, expected",
    [
        (2, 2, 6.0 / 9.0),
        (2, 4, 8.0 / 17.0),
        (1, 2, 0.8),
    ],
)
def test_purity_induced_rank(n_qubits, rank, expected) -> None:
    # Induced-measure mean purity E[Tr rho^2] = (d+K)/(dK+1), K = rank.
    rho = DensityMatrix.induced(
        n_qubits=n_qubits, n_samples=4000, rank=rank, random_key=KEY
    )
    purity = float(_purity(rho).mean())
    assert math.isclose(purity, expected, abs_tol=0.02), (
        f"Induced mean purity {purity} deviates from {expected}"
    )


@pytest.mark.unittest
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_induced_rank_one_is_pure(n_qubits) -> None:
    # rank=1 yields rank-1 (pure) states with Tr rho^2 = 1 exactly per sample.
    rho = DensityMatrix.induced(n_qubits=n_qubits, n_samples=8, rank=1, random_key=KEY)
    assert jnp.allclose(_purity(rho), 1.0, atol=1e-8), "rank=1 states are not pure"


@pytest.mark.unittest
@pytest.mark.parametrize("n_qubits", [1, 2])
def test_bures_purity(n_qubits) -> None:
    # Bures mean purity E[Tr rho^2] = (5N^2 + 1) / (2N(N^2 + 2)), N = d
    # (Osipov-Sommers-Zyczkowski, arXiv:1004.1655), verified empirically.
    d = 2**n_qubits
    expected = (5 * d**2 + 1) / (2 * d * (d**2 + 2))
    rho = DensityMatrix.bures(n_qubits=n_qubits, n_samples=4000, random_key=KEY)
    purity = float(_purity(rho).mean())
    assert math.isclose(purity, expected, abs_tol=0.02), (
        f"Bures mean purity {purity} deviates from {expected}"
    )


@pytest.mark.unittest
@pytest.mark.parametrize(
    "n_qubits, eigenvalues",
    [
        (1, [0.7, 0.3]),
        (2, [0.5, 0.3, 0.15, 0.05]),
    ],
)
def test_eigen_fixed_spectrum(n_qubits, eigenvalues) -> None:
    eigenvalues = jnp.array(eigenvalues)
    rho = DensityMatrix.eigen(
        n_qubits=n_qubits, n_samples=8, eigenvalues=eigenvalues, random_key=KEY
    )
    spectrum = jnp.sort(jnp.linalg.eigvalsh(rho).real, axis=-1)
    expected = jnp.sort(eigenvalues)
    assert jnp.allclose(spectrum, expected, atol=1e-8), (
        "Sampled spectrum does not match the supplied eigenvalues"
    )


@pytest.mark.unittest
def test_eigen_validation_errors() -> None:
    # Wrong length
    with pytest.raises(ValueError):
        DensityMatrix.eigen(n_qubits=2, eigenvalues=jnp.array([1.0]))
    # Negative entry
    with pytest.raises(ValueError):
        DensityMatrix.eigen(n_qubits=1, eigenvalues=jnp.array([1.2, -0.2]))
    # Does not sum to 1
    with pytest.raises(ValueError):
        DensityMatrix.eigen(n_qubits=1, eigenvalues=jnp.array([0.5, 0.4]))
