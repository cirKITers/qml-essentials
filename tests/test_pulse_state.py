import pytest
import jax
import jax.numpy as jnp

from qml_essentials.pulses import PulseGates, PulseInformation
from qml_essentials.yaqsi import Yaqsi, Script

jax.config.update("jax_enable_x64", True)


def assert_default_pulse_state():
    assert PulseInformation.get_envelope() == PulseInformation.DEFAULT_ENVELOPE
    assert PulseInformation.get_rwa() is PulseInformation.DEFAULT_RWA
    assert PulseInformation.get_frame() == PulseInformation.DEFAULT_FRAME
    assert PulseGates._active_envelope == PulseInformation.DEFAULT_ENVELOPE
    assert PulseGates._active_rwa is PulseInformation.DEFAULT_RWA
    assert PulseGates._active_frame == PulseInformation.DEFAULT_FRAME


def test_snapshot_restore_restores_config_and_leaf_params():
    snapshot = PulseInformation.snapshot_state()
    original_rx = PulseInformation.RX.params

    PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
    PulseInformation.RX.params = jnp.ones_like(PulseInformation.RX.params) * 0.123

    PulseInformation.restore_state(snapshot)

    assert PulseInformation.get_envelope() == snapshot.envelope
    assert PulseInformation.get_rwa() is snapshot.rwa
    assert PulseInformation.get_frame() == snapshot.frame
    assert PulseGates._active_envelope == snapshot.envelope
    assert PulseGates._active_rwa is snapshot.rwa
    assert PulseGates._active_frame == snapshot.frame
    assert jnp.allclose(PulseInformation.RX.params, original_rx)


def test_preserve_state_restores_after_exception():
    snapshot = PulseInformation.snapshot_state()

    with pytest.raises(RuntimeError, match="boom"):
        with PulseInformation.preserve_state():
            PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
            PulseInformation.RY.params = (
                jnp.ones_like(PulseInformation.RY.params) * 0.456
            )
            raise RuntimeError("boom")

    assert PulseInformation.get_envelope() == snapshot.envelope
    assert PulseInformation.get_rwa() is snapshot.rwa
    assert PulseInformation.get_frame() == snapshot.frame
    assert jnp.allclose(PulseInformation.RY.params, snapshot.leaf_params["RY"])


def test_00_autouse_fixture_allows_unrestored_mutation():
    PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
    PulseInformation.RX.params = jnp.ones_like(PulseInformation.RX.params) * 0.789

    assert PulseInformation.get_envelope() == "gaussian"
    assert PulseInformation.get_rwa() is False
    assert PulseInformation.get_frame() == "lab"


def test_01_autouse_fixture_restores_after_previous_test():
    assert_default_pulse_state()


def test_set_envelope_evicts_stale_solver_cache():
    """Regression test for the order-dependent fidelity failures.

    Building an evolution under one envelope cached a compiled XLA
    program keyed on coefficient-function code object identity.
    Switching the envelope rebuilt the coefficient functions, but the
    cache key (``id(fn.__code__)``) could collide with a freshly
    allocated code object, returning the stale program for a different
    pulse shape and silently degrading fidelity.

    With cache invalidation in place, the cache must be empty after a
    state change, and a freshly evaluated fidelity for the current
    envelope must be perfect.
    """

    def pulse_circuit(w, pp):
        PulseGates.RX(w, wires=0, pulse_params=pp)

    def target_circuit(w):
        from qml_essentials.operations import RX as OpRX

        OpRX(w, wires=0)

    # Prime the cache under a different envelope.
    PulseInformation.set_envelope("gaussian")
    Script(pulse_circuit, n_qubits=1).execute(
        type="state", args=(jnp.pi / 4, PulseInformation.RX.params)
    )
    assert len(Yaqsi._evolve_solver_cache) >= 1

    # Switch back to the default envelope.  Stale entries that referenced
    # the gaussian coefficient functions must be evicted so they cannot
    # be returned for the new (drag) coefficient functions.
    PulseInformation.set_envelope(PulseInformation.DEFAULT_ENVELOPE)
    assert len(Yaqsi._evolve_solver_cache) == 0

    pulse_script = Script(pulse_circuit, n_qubits=1)
    target_script = Script(target_circuit, n_qubits=1)
    state_pulse = pulse_script.execute(
        type="state", args=(jnp.pi / 2, PulseInformation.RX.params)
    )
    state_target = target_script.execute(type="state", args=(jnp.pi / 2,))
    fidelity = float(jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2)
    assert jnp.isclose(fidelity, 1.0, atol=1e-2), (
        f"Stale solver cache contaminated fidelity: {fidelity}"
    )
