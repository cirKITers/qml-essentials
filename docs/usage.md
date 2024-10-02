# Usage

Central component of our package is the Fourier model which you can import with 
```python
from qml_essentials.model import Model
```

In the simplest scenario, one would instantiate such a model with $2$ qubits and a single layer using the "Hardware Efficient" ansatz by:
```python
model = Model(
    n_qubits=2
    n_layers=1
    circuit_type="HardwareEfficient"
)
```

You can take a look at your model, by simply calling
```python
print(model)
```

You can also provide a custom circuit, by instantiating from the `Circuit` class in `qml_essentials.ansaetze.Circuit`.
See page ["Ansaetze"](ansaetze.md) for more details.

Calling the model without any (`None`) values for the `params` and `inputs` argument, will implicitly call the model with the recently (or initial) parameters and `0`s as input.

In the following we will describe some concepts of this class.
For a more detailled reference on the methods and arguments that are available, please see the [references page](https://cirkiters.github.io/qml-essentials/references/#model).

## The essentials

There is much more to this package, than just providing a Fourier model.
You can calculate the [Expressibility](expressibility.md) or [Entangling Capability](entanglement.md) besides the [Coefficients](coefficients.md) which are unique to this kind of QML interpretation.
Also checkout the available [Ansaetze](ansaetze.md) that we provide with this package.

## Output Shape

The output shape is determined by the `output_qubit` argument, provided in the instantiation of the model.
When set to -1 all qubits are measured which will result in the shape being of size $n$ by default (depending on the execution type, see below).

If `force_mean` flag is set when calling the model, the output is averaged to a single value (while keeping the batch dimension).

## Execution Type

Our model be simulated in different ways by setting the `execution_type` property, when calling the model, to:
- `exp_val`: Returns the expectation value between $0$ and $1$
- `density`: Calculates the density matrix
- `probs`: Simulates the model with the number of shots, set by `model.shots`

## Noise

Noise can be added to the model by providing a `noise_params` argument, when calling the model, which is a dictionary with following keys
- `BitFlip`
- `PhaseFlip`
- `AmplitudeDamping`
- `PhaseDamping`
- `DepolarizingChannel`
with values between $0$ and $1$.

This will apply the corresponding noise in each layer with the provided factor.

## Caching

To speed up calculation, you can add `cache=True` when calling the model.
The result of the model call will then be stored in a numpy format in a folder `.cache`.
Each result is being identified by a md5 hash that is a representation of the following model properties:
- number of qubits
- number of layers
- ansatz
- data-reuploading flag
- parameters
- noise parameters
- execution type
- inputs
- output qubit(s)