# Pulses

Models can be run at the **pulse level**, where each gate is implemented as a time-dependent control pulse rather than an abstract unitary.
This provides a more fine grained access to the simulation of the underlying physical process.
While we provide a developer-oriented overview in this section, we would like to highlight [Tilmann's Bachelor's Thesis](https://doi.org/10.5445/IR/1000184129) if you want to have a more detailled read into pulse-level simulation and quantum Fourier models.

The pulse gates themselves, their envelopes, the ODE solver and the quantum optimal control tooling live in the [jaqsi simulator](https://cirkiters.github.io/jaqsi/pulses/).
This page covers how to drive them from a `Model`.

## Building Ansatzes in Pulse Mode

When building an ansatz in pulse mode (via a `Model`), the framework internally passes an array of ones as element-wise scalers for the optimized parameters.  
If `pulse_params` are provided for a model or gate, these are treated similarly as element-wise scalers to modify the default pulses. We again take advantage of the **kwargs and call:

```python
model(pulse_params=model.pulse_params * 1.5)
```

Here, input and params are inferred from the `Model` instance, and we scale all pulse parameters by a factor of 1.5.
Currently there is no way to change the raw values of pulse parameter through the model api directly.

Note that a `Model` selects the pulse level differently than an individual gate does.
A gate takes `gate_mode`, which is either `"unitary"` or `"pulse"`, whereas a model infers this per gate group from the pulse parameters you pass: `pulse_params` lowers the ansatz and state-preparation gates, `enc_pulse_params` lowers the input-encoding gates.
Passing neither keeps the whole circuit unitary.
The `gate_mode` argument of a model call is deprecated and only kept for backwards compatibility.

Similar to the input and standard parameters, we also support batching for the `pulse_params` argument, meaning that you can also pass a batched array of pulse parameters of e.g. size 2 to as follows:

```python
model(pulse_params=np.repeat(model.pulse_params, 2, axis=0))
``` 

## Pulse-Level Encoding

Passing `pulse_params` runs the ansatz and state-preparation gates at pulse level while the input-encoding gates are still applied as ideal unitaries.
The encoding gates are covered by `enc_pulse_params`: passing only those runs the encoding at pulse level and keeps the ansatz unitary, passing both runs everything at pulse level.
Pass only `enc_pulse_params` if you want to study the encoding in isolation, without the ansatz pulses contributing to the result.
The scalers behave identically to the pulse parameters of trainable unitaries.

```python
model = Model(n_qubits=2, n_layers=1, circuit_type="Hardware_Efficient")

# only the encoding gates run as pulses, the ansatz stays unitary
model(inputs=inputs, enc_pulse_params=model.enc_pulse_params)

# encoding and ansatz gates run as pulses
model(inputs=inputs, pulse_params=model.pulse_params, enc_pulse_params=model.enc_pulse_params)

# scale the encoding pulses explicitly
model(inputs=inputs, enc_pulse_params=model.enc_pulse_params * 1.5)
```

`enc_pulse_params` is batchable along its leading axis, analogous to `pulse_params`:

```python
model(inputs=inputs, enc_pulse_params=np.repeat(model.enc_pulse_params, 2, axis=0))
```

Each parameter set only lowers the gates it belongs to, so the group whose parameters you omit stays unitary.

Note that Golomb encoding and custom encoding callables have no pulse level representation, so passing `enc_pulse_params` raises a `ValueError` for them.


## Pulse Envelopes

Each pulse is shaped by an envelope; the available ones are `gaussian`, `square`, `cosine`, `drag`, `sech` and `general` (see `PulseEnvelope.available()`).
The envelope is selected globally via the `pulse_shape` argument of the `Model`, e.g. `Model(..., pulse_shape="drag")`.

For the underlying pulse parametrization, the ODE solver settings and how to tune pulse parameters with quantum optimal control, see the [jaqsi pulses documentation](https://cirkiters.github.io/jaqsi/pulses/).
