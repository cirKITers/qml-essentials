"""Tests for the matchgate / XY-brickwork ansaetze, the loss-variance harness,
and the Model custom-observables hook (added for the angle-encoded QFM migration).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qml_essentials.ansaetze import Ansaetze
from qml_essentials.algebra import (
    lie_closure_paulis,
    matchgate_basis,
    dim_so2n,
    g_purity_from_basis,
)
from qml_essentials import trainability
from jaqsi import gateset
from qml_essentials.model import Model


def _pauli(n, positions, letter):
    s = ["I"] * n
    for p in positions:
        s[p] = letter
    return "".join(s)


def _product_state(theta):
    """prod_k R_y(theta_k)|0>, qubit 0 leftmost."""
    psi = np.array([1.0 + 0j])
    for t in theta:
        psi = np.kron(psi, np.array([np.cos(t / 2), np.sin(t / 2)], dtype=complex))
    return psi


@pytest.mark.unittest
def test_matchgate_ansatz_param_counts_and_dla() -> None:
    """Matchgate (RZ+RXX) closes to so(2n) [dim n(2n-1)]; XY-brickwork (RXX+RYY)
    to so(n)+so(n) [dim n(n-1)]; layer widths are n+(n-1) and 2(n-1)."""
    for n in range(2, 7):
        assert Ansaetze.Matchgate.n_params_per_layer(n) == n + (n - 1)
        assert Ansaetze.XY_Brickwork.n_params_per_layer(n) == 2 * (n - 1)
    for n in range(2, 6):
        mg_gens = [_pauli(n, [k], "Z") for k in range(n)] + [
            _pauli(n, [k, k + 1], "X") for k in range(n - 1)
        ]
        xy_gens = [_pauli(n, [k, k + 1], "X") for k in range(n - 1)] + [
            _pauli(n, [k, k + 1], "Y") for k in range(n - 1)
        ]
        assert len(lie_closure_paulis(mg_gens)) == n * (2 * n - 1)
        assert len(lie_closure_paulis(xy_gens)) == n * (n - 1)


@pytest.mark.unittest
def test_loss_variance_matches_analytic_matchgate() -> None:
    """trainability.loss_variance -> P_g(rho)/dim g for the matchgate LASA."""
    for n in (4, 6):
        theta = np.random.default_rng(2).uniform(0.0, 2 * np.pi, n)
        pred = float(
            g_purity_from_basis(_product_state(theta), matchgate_basis(n))
        ) / dim_so2n(n)
        var, losses = trainability.loss_variance(
            Ansaetze.Matchgate.build,
            Ansaetze.Matchgate.n_params_per_layer(n),
            theta,
            6 * n,
            1500,
            jax.random.PRNGKey(7),
        )
        assert losses.shape == (1500,)
        assert 0.7 < var / pred < 1.4


@pytest.mark.unittest
def test_model_observables_hook() -> None:
    """observables= returns one expval per observable; the default PauliZ path is
    unchanged; the values match a state-based computation; a setter is available."""
    n = 4
    Id = np.eye(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    def kron(*ms):
        r = np.array([[1]], dtype=complex)
        for m in ms:
            r = np.kron(r, m)
        return r

    XXm, YYm = kron(Id, X, X, Id), kron(Id, Y, Y, Id)
    XX = gateset.PauliX(wires=1) @ gateset.PauliX(wires=2)
    YY = gateset.PauliY(wires=1) @ gateset.PauliY(wires=2)
    inp = jnp.array([[0.5]])

    m_def = Model(n_qubits=n, n_layers=1, circuit_type="Matchgate", data_reupload=False)
    assert np.asarray(m_def(inputs=inp, execution_type="expval")).reshape(-1).shape == (
        n,
    )

    m = Model(
        n_qubits=n,
        n_layers=1,
        circuit_type="Matchgate",
        data_reupload=False,
        observables=[XX, YY],
    )
    ev = np.asarray(m(inputs=inp, execution_type="expval")).reshape(-1)
    st = np.asarray(m(inputs=inp, execution_type="state")).reshape(-1)
    manual = np.array([(st.conj() @ (XXm @ st)).real, (st.conj() @ (YYm @ st)).real])
    assert ev.shape == (2,)
    assert np.max(np.abs(ev - manual)) < 1e-5

    m2 = Model(n_qubits=n, n_layers=1, circuit_type="Matchgate", data_reupload=False)
    m2.observables = [XX, YY]
    assert np.asarray(m2(inputs=inp, execution_type="expval")).reshape(-1).shape == (2,)


@pytest.mark.unittest
def test_model_expval_differentiable_under_jit() -> None:
    """value_and_grad + jit through Model.__call__ (the angle-encoded training path)."""
    n, depth = 4, 3
    XX = gateset.PauliX(wires=1) @ gateset.PauliX(wires=2)
    YY = gateset.PauliY(wires=1) @ gateset.PauliY(wires=2)
    dmask = np.broadcast_to(np.eye(n, dtype=bool), (depth, n, n)).copy()
    model = Model(
        n_qubits=n,
        n_layers=depth,
        circuit_type="XY_Brickwork",
        data_reupload=dmask,
        encoding=["RY"] * n,
        observables=[XX, YY],
    )
    Theta = jnp.asarray(np.random.default_rng(0).uniform(0, 2 * np.pi, (8, n)))
    y = jnp.asarray(np.random.default_rng(1).normal(size=8))

    def loss(W):
        return jnp.mean(
            (model(params=W, inputs=Theta, execution_type="expval").sum(-1) - y) ** 2
        )

    W0 = jax.random.uniform(
        jax.random.PRNGKey(2), model._params_shape, minval=0, maxval=2 * np.pi
    )
    lo, g = jax.jit(jax.value_and_grad(loss))(W0)
    assert np.isfinite(float(lo))
    assert g.shape == W0.shape
    assert float(jnp.linalg.norm(g)) > 0
