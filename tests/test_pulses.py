# flake8: noqa: E731
import os
os.environ["JAX_ENABLE_X64"] = "1"
import pennylane as qml
import pennylane.numpy as np
import pytest
from qml_essentials.model import Model


@pytest.mark.unittest
@pytest.mark.skip(reason="JAX migration required")
def test_pulse_model() -> None:
    model = Model(
        n_qubits=4,
        n_layers=2,
        circuit_type="Hardware_Efficient",
    )

    # setting test data
    domain = [-np.pi, np.pi]
    omegas = np.array([1, 2, 3, 4])
    coefficients = np.array([1, 1, 1, 1])
    n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))
    x = np.linspace(domain[0], domain[1], num=n_d)

    def f(x):
        return 1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))

    y = np.stack([f(sample) for sample in x])

    def cost_fct(params, pulse_params):
        y_hat = model(
            params=params,
            pulse_params=pulse_params,
            inputs=x,
            force_mean=True,
            gate_mode="pulse"
        )
        return np.mean((y_hat - y) ** 2)

    opt = qml.AdamOptimizer(stepsize=0.01)
    pulse_params_before = model.pulse_params.copy()
    (model.params, model.pulse_params), cost_val = opt.step_and_cost(
        cost_fct, model.params, model.pulse_params
    )
    pulse_params_after = model.pulse_params.copy()

    assert not np.allclose(
        pulse_params_before, pulse_params_after
    ), "pulse_params did not update during training"

    grads = qml.grad(cost_fct, argnum=1)(model.params, model.pulse_params)
    assert np.any(np.abs(grads) > 1e-6), "Gradient wrt pulse_params is too small"


# if __name__ == "__main__":
#     print("Starting test")
#     # test_pulse_model()
#     test_pulse_model_inference()
#     print("Test complete")
