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
model.draw(figure="mpl")
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

## Data-Reuploading

This idea is one of the core features of our framework and builds upon the work by [*Schuld et al. (2020)*](https://arxiv.org/abs/2008.08605).
Essentially it allows us to represent a quantum circuit as a truncated Fourier series which is a powerfull feature that enables the model to mimic arbitrary non-linear functions.
The number of frequencies that the model can represent is constrained by the number of data encoding steps within the circuit.

Typically, there is a reuploading step after each layer and on each qubit (`data_reupload=True`).
However, our package also allows you to specify and array with the number of rows representing the qubits and number of columns representing the layers.
Then a `1` means that encoding is applied at the corresponding position within the circuit.

In the following example, the model has two reuploading steps (`model.degree` = 2) although it would be capable of representing four frequencies:

```python
model = Model(
    n_qubits=2,
    n_layers=2,
    circuit_type="Hardware_Efficient",
    data_reupload=[[1, 0], [0, 1]],
)
```

Checkout the [*Coefficients*](coefficients.md) page for more details on how you can visualize such a model using tools from signal analysis.

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

If you want to visualize zero-valued encoding gates in the model, set `remove_zero_encoding` to `False` on instantiation.

## Output Shape

The output shape is determined by the `output_qubit` argument, provided in the instantiation of the model.
When set to -1 all qubits are measured which will result in the shape being of size $n$ by default (depending on the execution type, see below).

If `force_mean` flag is set when calling the model, the output is averaged to a single value (while keeping the batch/ input dimension).
This is usually helpful, if you want to perform a n-local measurement over all qubits where only the average over $n$ expecation values is of interest.

## Execution Type

Our model be simulated in different ways by setting the `execution_type` property, when calling the model, to:

- `expval`: Returns the expectation value between $0$ and $1$
- `density`: Calculates the density matrix
- `probs`: Simulates the model with the number of shots, set by `model.shots`

## Noise

Noise can be added to the model by providing a `noise_params` argument, when calling the model, which is a dictionary with following keys

- `BitFlip`
- `PhaseFlip`
- `AmplitudeDamping`
- `PhaseDamping`
- `Depolarizing`
- `StatePreparation`
- `Measurement`

with values between $0$ and $1$.
Additionally, a `GateError` can be applied, which controls the variance of a Gaussian distribution with zero mean applied on the input vector.

While `BitFlip`, `PhaseFlip`, `Depolarizing` and `GateError`s are applied on each gate, `AmplitudeDamping`, `PhaseDamping`, `StatePreparation` and `Measurement` are applied on the whole circuit.

Furthermore, `ThermalRelaxation` can be applied. 
Instead of the probability, the entry for this type of error consists of another dict with the keys:

- `t1`: The relative T1 relaxation time (a typical value might be $180\mathrm{us}$)
- `t2`: The relative T2 relaxation time (a typical value might be $100\mathrm{us}$)
- `t_factor`: The relative gate time factor (a typical value might be $0.018\mathrm{us}$)

The units can be ignored as we are only interested in relative times, above values might belong to some superconducting system.
Note that `t2` is required to be max. $2\times$`t1`.
Based on `t_factor` and the circuit depth the execution time is estimated, and therefore the influence of thermal relaxation over time.

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

## Quantikz Export

In addition to the printing the model to console and into a figure using matplotlib (thanks to Pennylane); our framework extends this functionality by allowing you to create nice [Quantikz](https://doi.org/10.48550/arXiv.1809.03842) figures that you can embedd in a Latex document :heart_eyes:.
This can be achieved by 

```python
model.draw(figure="tikz", gate_values=False)
```

![Tikz Circuit](circuit_tikz_light.png#only-light)
![Tikz Circuit](circuit_tikz_dark.png#only-dark)

If you want to see the actual gate values instead of variables, simply set `gate_values=True` which is also the default option.