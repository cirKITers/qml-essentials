import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
import pytest

from qml_essentials.model import Model
from qml_essentials.script import Script
from qml_essentials.operations import RX, RY, CX
from qml_essentials.math import (
    quantum_fisher_information,
    fubini_study_metric,
    fidelity,
    trace_distance,
    phase_difference,
    logm_v,
)

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
            lambda _p: jnp.ones((2, 3), dtype=jnp.complex128), jnp.array([0.1])
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


def test_qfi_jaqsi_circuit():
    """QFI and Fubini-Study metric for a circuit built directly with JAQSI."""

    def state_fn(theta):
        def circuit(t):
            RX(t[0], wires=0)
            RY(t[1], wires=1)
            CX(wires=[0, 1])

        return Script(circuit, n_qubits=2).execute(type="state", args=(theta,))

    theta = jnp.array([0.7, 1.3])
    F = quantum_fisher_information(state_fn, theta)
    g = fubini_study_metric(state_fn, theta)

    # RX and RY each contribute a QFI of 1 on independent qubits; the trailing
    # parameter-independent CX leaves the QFI invariant.
    assert jnp.allclose(F, jnp.eye(2), atol=1e-10)
    assert jnp.allclose(F, 4.0 * g, atol=1e-10)


def _hermitian_pd_2x2():
    """A fixed 2x2 Hermitian positive-definite matrix."""
    return jnp.array([[2.0, 0.5j], [-0.5j, 1.0]], dtype=jnp.complex128)


def _logm_reference(matrix):
    """Matrix logarithm via Hermitian eigendecomposition (independent of scipy)."""
    evals, evecs = jnp.linalg.eigh(matrix)
    log_evals = jnp.log(evals.astype(jnp.complex128))
    return evecs @ jnp.diag(log_evals) @ jnp.conj(evecs.T)


def test_logm_v_single():
    """logm_v of a single matrix matches the eigendecomposition reference."""
    M = _hermitian_pd_2x2()
    result = logm_v(M)
    assert jnp.allclose(result, _logm_reference(M), atol=1e-10)


def test_logm_v_batched():
    """logm_v applies the logarithm to each matrix in a batch."""
    M0 = _hermitian_pd_2x2()
    M1 = jnp.array([[1.5, 0.0], [0.0, 0.5]], dtype=jnp.complex128)
    batch = jnp.stack([M0, M1])
    result = logm_v(batch)
    assert result.shape == batch.shape
    assert jnp.allclose(result[0], _logm_reference(M0), atol=1e-10)
    assert jnp.allclose(result[1], _logm_reference(M1), atol=1e-10)


def test_logm_v_invalid_shape_raises():
    """A non-matrix input shape is rejected."""
    with pytest.raises(NotImplementedError):
        logm_v(jnp.array([1.0, 2.0]))


class TestFidelity:
    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "sv0,sv1,expected_val",
        [
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1.0, 0.0]),
                1.0,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([0.0, 1.0]),
                0.0,
            ),
            (
                jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j]),
                jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j]),
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "overlap"],
    )
    def test_fidelity_statevector(self, sv0, sv1, expected_val):
        """Fidelity of state vectors should match PennyLane."""
        result = fidelity(sv0, sv1)
        expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_statevector_batched(self):
        """Batched state-vector fidelity should match element-wise PennyLane results."""
        sv0_batch = jnp.array(
            [[1.0, 0.0], [0.0, 1.0], [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)]]
        )
        sv1_batch = jnp.array(
            [[1.0, 0.0], [1.0, 0.0], [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)]]
        )
        result = fidelity(sv0_batch, sv1_batch)
        for i in range(3):
            expected_i = qml.math.fidelity_statevector(
                np.array(sv0_batch[i]), np.array(sv1_batch[i])
            )
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "rho0,rho1,expected_val",
        [
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                1.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128),
                0.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.eye(2, dtype=jnp.complex128) / 2,
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "mixed"],
    )
    def test_fidelity_dm(self, rho0, rho1, expected_val):
        """Fidelity of density matrices should match PennyLane."""
        result = fidelity(rho0, rho1)
        expected = qml.math.fidelity(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_dm_batched(self):
        """Batched density-matrix fidelity should match element-wise PennyLane."""
        rho0_batch = jnp.array(
            [
                [[1, 0], [0, 0]],
                [[0.5, 0.5], [0.5, 0.5]],
                jnp.eye(2) / 2,
            ],
            dtype=jnp.complex128,
        )
        rho1_batch = jnp.array(
            [
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                jnp.eye(2) / 2,
            ],
            dtype=jnp.complex128,
        )
        result = fidelity(rho0_batch, rho1_batch)
        for i in range(3):
            expected_i = qml.math.fidelity(
                np.array(rho0_batch[i]), np.array(rho1_batch[i])
            )
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_dm_matches_statevector(self):
        """Density-matrix fidelity of pure states should equal state-vector fidelity."""
        sv0 = jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j])
        sv1 = jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j])
        rho0 = jnp.outer(sv0, jnp.conj(sv0))
        rho1 = jnp.outer(sv1, jnp.conj(sv1))
        f_sv = fidelity(sv0, sv1)
        f_dm = fidelity(rho0, rho1)
        assert jnp.allclose(f_sv, f_dm, atol=1e-8)

    @pytest.mark.unittest
    def test_fidelity_mismatched_raises(self):
        """Passing a vector and a matrix should raise ValueError."""
        sv = jnp.array([1.0, 0.0])
        dm = jnp.eye(2, dtype=jnp.complex128)
        with pytest.raises(ValueError, match="same kind"):
            fidelity(sv, dm)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "rho0,rho1,expected_val",
        [
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                0.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128),
                1.0,
            ),
            (
                jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
                jnp.eye(2, dtype=jnp.complex128) / 2,
                None,  # just check against PennyLane
            ),
        ],
        ids=["identical", "orthogonal", "mixed"],
    )
    def test_trace_distance(self, rho0, rho1, expected_val):
        """Trace distance should match PennyLane."""
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_trace_distance_batched(self):
        """Batched trace distance should match element-wise PennyLane."""
        batch0 = jnp.array(
            [jnp.eye(2) / 2, jnp.ones((2, 2)) / 2, jnp.array([[1, 0], [0, 0]])],
            dtype=jnp.complex128,
        )
        batch1 = jnp.array(
            [jnp.ones((2, 2)) / 2, jnp.ones((2, 2)) / 2, jnp.array([[1, 0], [0, 0]])],
            dtype=jnp.complex128,
        )
        result = trace_distance(batch0, batch1)
        expected = qml.math.trace_distance(np.array(batch0), np.array(batch1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_trace_distance_from_statevectors(self):
        """Trace distance computed from outer-product DMs should match PennyLane."""
        sv0 = jnp.array([0.2, jnp.sqrt(0.96)])
        sv1 = jnp.array([1.0, 0.0])
        rho0 = jnp.outer(sv0, jnp.conj(sv0))
        rho1 = jnp.outer(sv1, jnp.conj(sv1))
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_fidelity_2qubit_statevector(self):
        """Fidelity of 2-qubit state vectors should match PennyLane."""
        sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell state
        sv1 = jnp.array([1, 0, 0, 0], dtype=jnp.complex128)  # |00⟩
        result = fidelity(sv0, sv1)
        expected = qml.math.fidelity_statevector(np.array(sv0), np.array(sv1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    def test_trace_distance_2qubit(self):
        """Trace distance of 2-qubit density matrices should match PennyLane."""
        bell = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])
        rho0 = jnp.outer(bell, jnp.conj(bell))
        rho1 = jnp.eye(4, dtype=jnp.complex128) / 4  # maximally mixed
        result = trace_distance(rho0, rho1)
        expected = qml.math.trace_distance(np.array(rho0), np.array(rho1))
        assert jnp.allclose(result, expected, atol=1e-10)

    @pytest.mark.unittest
    @pytest.mark.parametrize(
        "sv0,sv1,expected_val",
        [
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1.0, 0.0]),
                0.0,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([-1.0, 0.0]),
                jnp.pi,
            ),
            (
                jnp.array([1.0, 0.0]),
                jnp.array([1j, 0.0]),
                jnp.pi / 2,
            ),
            (
                jnp.array([0.98753537 - 0.14925137j, 0.00746879 - 0.04941796j]),
                jnp.array([0.99500417 + 0.0j, 0.09983342 + 0.0j]),
                None,  # just check against vdot formula
            ),
        ],
        ids=["identical", "global_phase", "half_pi", "arbitrary"],
    )
    def test_phase_difference(self, sv0, sv1, expected_val):
        """Phase difference should match the vdot formula."""
        result = phase_difference(sv0, sv1)
        expected = jnp.angle(jnp.vdot(sv0, sv1))
        assert jnp.allclose(result, expected, atol=1e-10)
        if expected_val is not None:
            assert jnp.allclose(result, expected_val, atol=1e-10)

    @pytest.mark.unittest
    def test_phase_difference_batched(self):
        """Batched phase difference should match element-wise computation."""
        sv0_batch = jnp.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1 / jnp.sqrt(2), 1 / jnp.sqrt(2)],
            ]
        )
        sv1_batch = jnp.array(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1 / jnp.sqrt(2), -1 / jnp.sqrt(2)],
            ]
        )
        result = phase_difference(sv0_batch, sv1_batch)
        for i in range(3):
            inner = jnp.sum(jnp.conj(sv0_batch[i]) * sv1_batch[i])
            expected_i = jnp.angle(inner)
            assert jnp.allclose(result[i], expected_i, atol=1e-10)

    @pytest.mark.unittest
    def test_phase_difference_2qubit(self):
        """Phase difference of 2-qubit states should match the vdot formula."""
        sv0 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1 / jnp.sqrt(2)])  # Bell |Φ+⟩
        sv1 = jnp.array([1 / jnp.sqrt(2), 0, 0, 1j / jnp.sqrt(2)])  # phase on |11⟩
        result = phase_difference(sv0, sv1)
        expected = jnp.angle(jnp.vdot(sv0, sv1))
        assert jnp.allclose(result, expected, atol=1e-10)
