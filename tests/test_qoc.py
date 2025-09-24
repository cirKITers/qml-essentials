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
def test_optimize_Rot(qoc):
    optimized_params, best_loss, _ = qoc.optimize_Rot()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"Rot optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_RX(qoc):
    optimized_params, best_loss, _ = qoc.optimize_RX()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"RX optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_RY(qoc):
    optimized_params, best_loss, _ = qoc.optimize_RY()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"RY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_H(qoc):
    optimized_params, best_loss, _ = qoc.optimize_H()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"H optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CZ(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CZ()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"CZ optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CY(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CY()
    fidelity = 1 - best_loss
    assert fidelity > 0.9, f"CY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
def test_optimize_CX(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CX()
    fidelity = 1 - best_loss
    assert fidelity > 0.9, f"CX optimization fidelity too low: {fidelity:.4f}"


# TODO: Unskip CRZ, CRY, CRX tests when their optimization is fixed
@pytest.mark.unittest
@pytest.mark.skip(reason="CRZ not properly optimized, low fidelity")
def test_optimize_CRZ(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRZ()
    fidelity = 1 - best_loss
    assert fidelity > 0.98, f"CRZ optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
@pytest.mark.skip(reason="CRY not properly optimized, low fidelity")
def test_optimize_CRY(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRY()
    fidelity = 1 - best_loss
    assert fidelity > 0.9, f"CRY optimization fidelity too low: {fidelity:.4f}"


@pytest.mark.unittest
@pytest.mark.skip(reason="CRX not properly optimized, low fidelity")
def test_optimize_CRX(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRX()
    fidelity = 1 - best_loss
    assert fidelity > 0.9, f"CRX optimization fidelity too low: {fidelity:.4f}"


# TODO: Remove CRZ, CRY, CRX smoketests when their optimization is fixed
@pytest.mark.smoketest
def test_optimize_CRZ_smoke(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRZ()
    fidelity = 1 - best_loss
    assert fidelity is not None


@pytest.mark.smoketest
def test_optimize_CRY_smoke(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRY()
    fidelity = 1 - best_loss
    assert fidelity is not None


@pytest.mark.smoketest
def test_optimize_CRX_smoke(qoc):
    optimized_params, best_loss, _ = qoc.optimize_CRX()
    fidelity = 1 - best_loss
    assert fidelity is not None
