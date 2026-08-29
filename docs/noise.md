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
Each gate draws its own error, independently of the other gates in the circuit.
It's important to note that, depending on the flag set in `UnitaryGates.batch_gate_error`, the error of a given gate will be applied to the entire batch of parameters (all batch elements are affected in the same way) or drawn for each batch element individually (default).
This can be particularly usefull in a scenario where one would like to apply noise e.g. only on the encoding gates but wants to change them all uniformly.
An example of this is provided in the following code:

```python
from jaqsi.gates import UnitaryGates

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

## Randomness under JAX transformations

Gate errors and shot sampling are the only parts of the simulation that draw random numbers at runtime as the decoherent channels above are deterministic maps on the density matrix.

By default these draws use the model's internal random key, which is advanced on every call.
Inside a JAX transformation this does not work: a jitted function is traced once, so the key is read at trace time and the compiled function replays the same noise realization on every call.
To get fresh randomness, pass a key explicitly and advance it outside the transformation with `next_key`:

```python
train_step = jax.jit(lambda params, key: cost(model(params=params, inputs=inputs, random_key=key)))

for _ in range(n_steps):
    loss = train_step(params, model.next_key())
```

Since the key is an argument rather than a constant, this does not trigger recompilation.
A warning is raised when a stochastic model is called inside a transformation without an explicit key.
