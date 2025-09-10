import jax.numpy as jnp
from qml_essentials.qoc import QOC
from qml_essentials import ansaetze
import pytest
import logging
import jax
jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)


# @pytest.fixture(autouse=True)
# def reset_pulse_mgr():
#     ansaetze.Gates._pulse_mgr = None


@pytest.mark.unittest
def test_optimize_RX():
    qoc = QOC()
    optimized_params, best_loss, _ = qoc.optimize_RX(
        w=jnp.pi, init_pulse_params=jnp.array([1.0, 15.0, 1.0])
    )
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"RX optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_RY():
    qoc = QOC()
    optimized_params, best_loss, _ = qoc.optimize_RY(
        w=jnp.pi, init_pulse_params=jnp.array([1.0, 15.0, 1.0])
    )
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"RY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_H():
    qoc = QOC()
    optimized_params, best_loss, _ = qoc.optimize_H(
        init_pulse_params=jnp.array([1.0, 15.0, 1.0])
    )
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"H optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CZ():
    qoc = QOC()
    optimized_params, best_loss, _ = qoc.optimize_CZ(
        init_pulse_params=jnp.array([0.975])
    )
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"CZ optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CNOT():
    qoc = QOC()
    optimized_params, best_loss, _ = qoc.optimize_CNOT(
        init_pulse_params=jnp.array([1.0, 15.0, 1.0, 1.0]),
    )
    fidelity = 1 - best_loss
    assert fidelity > 0.9, f"CNOT optimization fidelity too low: {fidelity:.4f}"
