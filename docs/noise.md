# Noise

In general, all of the Ansatzes that are implemented in this package allow the additional input below which is a dictionary containing all the noise parameters of the circuit (here all with probability $0.0$):
```python
noise_params = {
    "BitFlip": 0.0,
    "PhaseFlip": 0.0,
    "AmplitudeDamping": 0.0,
    "PhaseDamping": 0.0,
    "Depolarizing": 0.0,
    "MultiQubitDepolarizing": 0.0,
}
```

Providing this optional input will apply the corresponding noise to the model where the Bit Flip, Phase Flip, Depolarizing and Two-Qubit Depolarizing Channels are applied after each gate and the Amplitude and Phase Damping are applied at the end of the circuit.
For more information on the available noise types, have a look [here](usage.md#noise).

To demonstrate this, let's recall the custom ansatz `MyHardwareEfficient` defined in [Custom Ansatz](ansaetze.md#custom_ansatz) and extend the model's usage:

```python
model(
    model.params,
    inputs=None,
    execution_type="density",
    noise_params={
        "BitFlip": 0.01,
        "PhaseFlip": 0.02,
        "AmplitudeDamping": 0.03,
        "PhaseDamping": 0.04,
        "Depolarizing": 0.05,
        "MultiQubitDepolarizing": 0.06
})
```

In addition to these decoherent errors, we can also apply a `GateError` which affects each parameterized gate as $w = w + \mathcal{N}(0, \epsilon)$, where $\sqrt{\epsilon}$ is the standard deviation of the noise, specified by the `GateError` key in the `noise_params` argument.
It's important to note that, depending on the flag set in `Ansaetze.UnitaryGates.batch_gate_error`, the error will be applied to the entire batch of parameters (all parameters are affected in the same way) or to each parameter individually (default).
This can be particularly usefull in a scenario where one would like to apply noise e.g. only on the encoding gates but wants to change them all uniformly.
An example of this is provided in the following code:

```python
from qml_essentials.gates import UnitaryGates

UnitaryGates.batch_gate_error = False
model(
    ...
    noise_params={
        "GateError": 0.01,
    }
)

def pqc_noise_free(*args, **kwargs):
    kwargs["noise_params"] = None
    return pqc(*args, **kwargs)
model.pqc = pqc_noise_free
```

> **Note:** When using a noisy circuit, make sure to run the model with the `density` execution type.
