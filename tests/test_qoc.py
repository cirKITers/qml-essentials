from qml_essentials.qoc import QOC
import pytest
import logging
import jax

jax.config.update("jax_enable_x64", True)


logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def qoc():
    """Return a single QOC instance for all tests in this module."""
    return QOC(file_dir=None)


@pytest.mark.unittest
@pytest.mark.skip(reason="Rot not implemented")
def test_optimize_Rot(qoc):
    optimize_1q = qoc.optimize("default.qubit", wires=1)
    optimized_params, loss_history = optimize_1q(qoc.create_Rot)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"Rot optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_RX(qoc):
    optimize_1q = qoc.optimize("default.qubit", wires=1)
    optimized_params, loss_history = optimize_1q(qoc.create_RX)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"RX optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_RY(qoc):
    optimize_1q = qoc.optimize("default.qubit", wires=1)
    optimized_params, loss_history = optimize_1q(qoc.create_RY)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"RY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_H(qoc):
    optimize_1q = qoc.optimize("default.qubit", wires=1)
    optimized_params, loss_history = optimize_1q(qoc.create_H)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"H optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CZ(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CZ)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"CZ optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CY(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CY)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.8, f"CY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CX(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CX)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.8, f"CX optimization fidelity too low: {fidelity:.4f}"


# TODO: Unskip CRZ, CRY, CRX tests when their optimization is fixed
@pytest.mark.unittest
@pytest.mark.skip(reason="CRZ not properly optimized, low fidelity")
def test_optimize_CRZ(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRZ)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.98, f"CRZ optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
@pytest.mark.skip(reason="CRY not properly optimized, low fidelity")
def test_optimize_CRY(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRY)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.9, f"CRY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
@pytest.mark.skip(reason="CRX not properly optimized, low fidelity")
def test_optimize_CRX(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRX)()
    fidelity = 1 - min(loss_history)
    assert fidelity > 0.9, f"CRX optimization fidelity too low: {fidelity:.4f}"


# TODO: Remove CRZ, CRY, CRX smoketests when their optimization is fixed
@pytest.mark.smoketest
def test_optimize_CRZ_smoke(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRZ)()
    fidelity = 1 - min(loss_history)
    assert fidelity is not None


@pytest.mark.smoketest
def test_optimize_CRY_smoke(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRY)()
    fidelity = 1 - min(loss_history)
    assert fidelity is not None


@pytest.mark.smoketest
def test_optimize_CRX_smoke(qoc):
    optimize_2q = qoc.optimize("default.qubit", wires=2)
    optimized_params, loss_history = optimize_2q(qoc.create_CRX)()
    fidelity = 1 - min(loss_history)
    assert fidelity is not None
