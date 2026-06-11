import jax
import jax.numpy as jnp
import pytest

from qml_essentials.model import Model
from qml_essentials.math import quantum_fisher_information, fubini_study_metric

jax.config.update("jax_enable_x64", True)


def _ry_state(theta):
    """Single-qubit RY state |psi> = cos(t/2)|0> + sin(t/2)|1>."""
    t = theta[0]
    return jnp.array([jnp.cos(t / 2), jnp.sin(t / 2)], dtype=jnp.complex128)


def _two_ry_state(theta):
    """Product state of two independent RY rotations."""
    a, b = theta[0], theta[1]
    q0 = jnp.array([jnp.cos(a / 2), jnp.sin(a / 2)], dtype=jnp.complex128)
    q1 = jnp.array([jnp.cos(b / 2), jnp.sin(b / 2)], dtype=jnp.complex128)
    return jnp.kron(q0, q1)


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.3, jnp.pi / 2])
def test_qfi_single_ry(theta):
    """QFI of a single RY rotation is the 1x1 matrix [[1]] for any angle."""
    F = quantum_fisher_information(_ry_state, jnp.array([theta]))
    assert F.shape == (1, 1)
    assert jnp.allclose(F, 1.0, atol=1e-8)


def test_qfi_two_independent_ry():
    """QFI of two independent RY rotations is the identity (no correlations)."""
    F = quantum_fisher_information(_two_ry_state, jnp.array([0.7, 1.3]))
    assert F.shape == (2, 2)
    assert jnp.allclose(F, jnp.eye(2), atol=1e-8)


def test_qfi_pure_mixed_consistency():
    """The mixed (SLD) formula reduces to the pure one on rho = |psi><psi|."""
    theta = jnp.array([0.7, 1.3])

    def rho_fn(p):
        psi = _two_ry_state(p)
        return jnp.outer(psi, jnp.conj(psi))

    F_pure = quantum_fisher_information(_two_ry_state, theta)
    F_mixed = quantum_fisher_information(rho_fn, theta)
    assert F_mixed.shape == (2, 2)
    assert jnp.allclose(F_pure, F_mixed, atol=1e-8)


def test_qfi_symmetric_and_psd():
    """The QFI is real, symmetric and positive semi-definite."""
    F = quantum_fisher_information(_two_ry_state, jnp.array([0.4, 2.1]))
    assert jnp.allclose(F, F.T, atol=1e-8)
    assert jnp.min(jnp.linalg.eigvalsh(F)) >= -1e-8


def test_qfi_invalid_state_shape():
    """A non-square 2D output is neither a state vector nor a density matrix."""
    with pytest.raises(ValueError):
        quantum_fisher_information(
            lambda p: jnp.ones((2, 3), dtype=jnp.complex128), jnp.array([0.1])
        )


def test_qfi_model_state():
    """End-to-end pure-state QFI differentiated through the JAQSI model."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    model.execution_type = "state"

    F = quantum_fisher_information(lambda p: model(params=p), model.params)

    P = model.params.size
    assert F.shape == (P, P)
    assert jnp.allclose(F, F.T, atol=1e-7)
    assert jnp.min(jnp.linalg.eigvalsh(F)) >= -1e-6


@pytest.mark.parametrize("theta", [0.0, 0.7, 1.3, jnp.pi / 2])
def test_fubini_study_single_ry(theta):
    """Fubini-Study metric of a single RY is QFI/4 = [[0.25]] for any angle."""
    g = fubini_study_metric(_ry_state, jnp.array([theta]))
    assert g.shape == (1, 1)
    assert jnp.allclose(g, 0.25, atol=1e-8)


def test_fubini_study_qfi_relation():
    """The pure-state QFI is four times the Fubini-Study metric."""
    theta = jnp.array([0.4, 2.1])
    g = fubini_study_metric(_two_ry_state, theta)
    F = quantum_fisher_information(_two_ry_state, theta)
    assert jnp.allclose(F, 4.0 * g, atol=1e-8)


def test_fubini_study_rejects_density():
    """The Fubini-Study metric is undefined for density matrices."""

    def rho_fn(p):
        psi = _two_ry_state(p)
        return jnp.outer(psi, jnp.conj(psi))

    with pytest.raises(ValueError):
        fubini_study_metric(rho_fn, jnp.array([0.7, 1.3]))


def test_fubini_study_model_state():
    """End-to-end Fubini-Study metric differentiated through the JAQSI model."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    model.execution_type = "state"
    params = model.params

    g = fubini_study_metric(lambda p: model(params=p), params)
    F = quantum_fisher_information(lambda p: model(params=p), params)

    P = params.size
    assert g.shape == (P, P)
    assert jnp.allclose(F, 4.0 * g, atol=1e-7)


def test_qfi_model_density():
    """End-to-end mixed-state QFI for a noisy (density-matrix) model."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    model.execution_type = "density"

    F = quantum_fisher_information(
        lambda p: model(params=p, noise_params={"BitFlip": 0.1}), model.params
    )

    P = model.params.size
    assert F.shape == (P, P)
    assert jnp.allclose(F, F.T, atol=1e-7)
    assert jnp.min(jnp.linalg.eigvalsh(F)) >= -1e-6
