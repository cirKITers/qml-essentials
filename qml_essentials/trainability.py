"""Empirical loss-variance (barren-plateau) diagnostics over random circuits.

Estimates ``Var_W[<O>]`` for a fixed input state evolved by a random
parameterised circuit, sampling the circuit parameters ``W ~ U[0, 2pi)``.  This
is the empirical left-hand side of the Ragone/Fontana LASA identity
``Var_W[<O>] = sum_j P_j(rho) P_j(O) / dim g_j`` used to certify trainability of
polynomial-DLA models; it complements the analytic vehicle
:func:`qml_essentials.algebra.g_purity_from_basis`.

The input is either an angle-encoded product state ``prod_k R_y(theta_k)|0>`` (pass
``theta``) or an explicit statevector (pass ``init_state``, e.g. a Haar-random
input), and the ansatz is any layer that writes onto the active jaqsi tape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qml_essentials.gates import Gates
from qml_essentials import operations as op
from qml_essentials import jaqsi as js
from qml_essentials.ansaetze import Ansaetze


def ansatz_layer(circuit_type: str):
    """Return ``(layer_fn, n_params_per_layer)`` for a built-in ``Ansaetze`` template.

    ``layer_fn(p, n)`` applies one layer of ``circuit_type`` (e.g. ``"Matchgate"``,
    ``"Strongly_Entangling"``) onto the active jaqsi tape, and
    ``n_params_per_layer(n)`` gives the length of ``p``.
    """
    cls = getattr(Ansaetze, circuit_type)

    def layer(params, n):
        cls.build(params, n)

    return layer, (lambda n: cls.n_params_per_layer(n))


def loss_variance(
    layer_fn,
    n_params_per_layer: int,
    theta: np.ndarray | None,
    depth: int,
    n_samples: int,
    key,
    obs=None,
    out_qubit: int | None = None,
    shots: int | None = None,
    init_state: np.ndarray | None = None,
):
    """Empirical ``Var_W[<O>]`` and the raw loss values.

    Args:
        layer_fn: applies one ansatz layer onto the tape, ``layer_fn(p, n)``.
        n_params_per_layer: length of ``p`` consumed per layer.
        theta: fixed angle configuration, shape ``(n,)``, angle-encoded as a
            product state ``prod_k R_y(theta_k)|0>``.  Ignored if ``init_state``
            is given.
        depth: number of ansatz layers (circuit-side mixing / 2-design depth).
        n_samples: number of random ``W`` draws.
        key: JAX PRNG key.
        obs: list of observables whose summed expectation is the loss; defaults
            to a single bulk-qubit ``[PauliZ(out_qubit)]`` (a matchgate readout).
        out_qubit: measured qubit for the default observable (default ``n // 2``).
        shots: finite shots for the expectation (``None`` -> exact).
        init_state: optional input statevector ``(2**n,)``; bypasses the ``R_y``
            product encoding, so ``theta`` is ignored when given (e.g. to inject
            a Haar-random small-g-purity input).

    Returns:
        ``(variance: float, losses: np.ndarray of shape (n_samples,))``.
    """
    if init_state is not None:
        init_state = jnp.asarray(init_state)
        n = int(round(float(np.log2(init_state.shape[-1]))))
    else:
        theta = jnp.asarray(theta, dtype=float)
        n = int(theta.shape[0])
    i_out = n // 2 if out_qubit is None else out_qubit
    if obs is None:
        obs = [op.PauliZ(wires=i_out)]

    # Separate streams for the parameter draws and the shot noise, so the two
    # sources of randomness stay independent.
    param_key, shot_key = jax.random.split(key)
    W = jax.random.uniform(
        param_key, (n_samples, depth, n_params_per_layer), minval=0.0, maxval=2 * np.pi
    )

    if init_state is None:

        def circ(params, th):
            for q in range(n):
                Gates.RY(th[q], wires=q)
            for d in range(depth):
                layer_fn(params[d], n)

        vals = js.Script(f=circ, n_qubits=n).execute(
            type="expval",
            obs=obs,
            args=(W, theta),
            in_axes=(0, None),
            shots=shots,
            key=shot_key,
        )
    else:

        def circ(params):  # init_state already encodes the input; apply only U(W)
            for d in range(depth):
                layer_fn(params[d], n)

        vals = js.Script(f=circ, n_qubits=n).execute(
            type="expval",
            obs=obs,
            args=(W,),
            in_axes=(0,),
            shots=shots,
            key=shot_key,
            initial_state=init_state,
        )
    vals = np.asarray(vals)
    if vals.ndim > 1 and len(obs) > 1:  # loss = <sum_i O_i>
        vals = vals.sum(axis=-1)
    vals = vals.reshape(-1)
    return float(np.var(vals, ddof=1)), vals
