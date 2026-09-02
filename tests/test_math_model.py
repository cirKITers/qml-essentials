"""End-to-end tests of the jaqsi quantum-geometry metrics on a Model.

The metrics themselves are tested in jaqsi; these tests cover differentiating
them through a full qml-essentials Model.
"""

import jax
import jax.numpy as jnp

from jaqsi.math import quantum_fisher_information, fubini_study_metric

from qml_essentials.model import Model

jax.config.update("jax_enable_x64", True)


def test_qfi_model_state():
    """End-to-end pure-state QFI differentiated through the model."""
    model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")
    model.execution_type = "state"

    F = quantum_fisher_information(lambda p: model(params=p), model.params)

    P = model.params.size
    assert F.shape == (P, P)
    assert jnp.allclose(F, F.T, atol=1e-7)
    assert jnp.min(jnp.linalg.eigvalsh(F)) >= -1e-6


def test_fubini_study_model_state():
    """End-to-end Fubini-Study metric differentiated through the model."""
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
