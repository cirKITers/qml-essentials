# Usage

Central component of our package is the Fourier model which you can import with 
```python
from qml_essentials.model import Model
```

In the simplest scenario, one would instantiate such a model with $2$ qubits and a single layer using the "Hardware Efficient" ansatz by:
```python
model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Hardware_Efficient",
)
```

You can take a look at your model, by simply calling
```python
model.draw(figure=True)
```

![Hardware Efficient Ansatz](hae_light.png#only-light)
![Hardware Efficient Ansatz](hae_dark.png#only-dark)

Looks good to you? :eyes: Head over to the [*Training*](training.md) page for **getting started** with an easy example :rocket:

Calling the model without any (`None`) values for the `params` and `inputs` argument, will implicitly call the model with the recently (or initial) parameters and `0`s as input.

In the following we will describe some concepts of the `Model` class.
For a more detailled reference on the methods and arguments that are available, please see the [references page](https://cirkiters.github.io/qml-essentials/references/#model).

## The essentials

There is much more to this package, than just providing a Fourier model.
You can calculate the [Expressibility](expressibility.md) or [Entangling Capability](entanglement.md) besides the [Coefficients](coefficients.md) which are unique to this kind of QML interpretation.
You can also provide a custom circuit, by instantiating from the `Circuit` class in `qml_essentials.ansaetze.Circuit`.
See page [*Ansaetze*](ansaetze.md) for more details and a list of available Ansatzes that we provide with this package.

## Parameter Initialization

The initialization strategy can be set when instantiating the model with the `initialization` argument.

The default strategy is "random" which will result in random initialization of the parameters using the domain specified in the `initialization_domain` argument.
Other options are:
- "zeros": All parameters are initialized to $0$
- "zero-controlled": All parameters are initialized to randomly except for the angles of the controlled rotations which are initialized to $0$
- "pi-controlled": All parameters are initialized to randomly except for the angles of the controlled rotations which are initialized to $\\pi$
- "pi": All parameters are initialized to $\\pi$

The `initialize_params` method provides the option to re-initialise the parameters after model instantiation using either the previous configuration or a different strategy.

## Encoding

The encoding can be set when instantiating the model with the `encoding` argument.

The default encoding is "RX" which will result in a single RX rotation per qubit.
Other options are:
- Any callable such as `Gates.RX`
- A list of callables such as `[Gates.RX, Gates.RY]`
- A string such as `"RX"` that will result in a single RX rotation per qubit
- A list of strings such as `["RX", "RY"]` that will result in a RX and RY rotation per qubit

See page [*Ansaetze*](ansaetze.md) for more details regarding the `Gates` class.
If a list of encodings is provided, the input is assumed to be multi-dimensional.
Otherwise multiple inputs are treated as batches of inputs.

## Output Shape

The output shape is determined by the `output_qubit` argument, provided in the instantiation of the model.
When set to -1 all qubits are measured which will result in the shape being of size $n$ by default (depending on the execution type, see below).

If `force_mean` flag is set when calling the model, the output is averaged to a single value (while keeping the batch/ input dimension).
This is usually helpful, if you want to perform a n-local measurement over all qubits where only the average over $n$ expecation values is of interest.

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
