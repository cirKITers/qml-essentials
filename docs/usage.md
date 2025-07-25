# Usage

Central component of our package is the Fourier model which you can import with 
```python
from qml_essentials.model import Model
```

In the simplest scenario, one would instantiate such a model with $4$ qubits and a single layer using the "Hardware Efficient" ansatz by:
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

![Hardware Efficient Ansatz](figures/hae_light.png#only-light)
![Hardware Efficient Ansatz](figures/hae_dark.png#only-dark)

Looks good to you? :eyes: Head over to the [*Training*](training.md) page for **getting started** with an easy example, where we also show how to implement **trainable frequencies** :rocket:

Note that calling the model without any (`None`) values for the `params` and `inputs` argument, will implicitly call the model with the recently (or initial) parameters and `0`s as input.
I.e. simply running the following
```python
model()
```
will return the combined expectation value of a n-local measurement (`output_qubit=-1` is default). 

In the following we will describe some concepts of the `Model` class.
For a more detailled reference on the methods and arguments that are available, please see the [references page](https://cirkiters.github.io/qml-essentials/references/#model).

## The essentials

There is much more to this package, than just providing a Fourier model.
You can calculate the [Expressibility](expressibility.md) or [Entangling Capability](entanglement.md) besides the [Coefficients](coefficients.md) which are unique to this kind of QML interpretation.
You can also provide a custom circuit, by instantiating from the `Circuit` class in `qml_essentials.ansaetze.Circuit`.
See page [*Ansaetze*](ansaetze.md) for more details and a list of available Ansatzes that we provide with this package.

## Data-Reuploading

This idea is one of the core features of our framework and builds upon the work by [*Schuld et al. (2020)*](https://doi.org/10.48550/arXiv.2008.08605).
Essentially it allows us to represent a quantum circuit as a truncated Fourier series which is a powerfull feature that enables the model to mimic arbitrary non-linear functions.
The number of frequencies that the model can represent is constrained by the number of data encoding steps within the circuit.

Typically, there is a reuploading step after each layer and on each qubit (`data_reupload=True`).
However, our package also allows you to specify and array with the number of rows representing the qubits and number of columns representing the layers.
Then a `True` means that encoding is applied at the corresponding position within the circuit.

In the following example, the model has two reuploading steps (`model.degree` = 2) although it would be capable of representing four frequencies:

```python
model = Model(
    n_qubits=2,
    n_layers=2,
    circuit_type="Hardware_Efficient",
    data_reupload=[[True, False], [False, True]],
)
```

Checkout the [*Coefficients*](coefficients.md) page for more details on how you can visualize such a model using tools from signal analysis.
If you want to encode multi-dimensional data (checkout the [*Encoding*](usage.md#encoding) section on how to do that), you can specify another dimension in the `data_reupload` argument (which just extents naturally).
```python
model = Model(
    n_qubits=2,
    n_layers=2,
    circuit_type="Hardware_Efficient",
    data_reupload=[[[0, 1], [1, 1]], [[1, 1], [0, 1]]],
)
```
Now, the first input will have two frequencies (`sum([0,1,1,0]) = 2`), and the second input will have four frequencies (`sum([1,1,1,1]) = 4`).
Of course, this is just a rule of thumb and can vary depending on the exact encoding strategy.

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

- A string such as `"RX"` that will result in a single RX rotation per qubit
- A list of strings such as `["RX", "RY"]` that will result in a sequential RX and RY rotation per qubit
- Any callable such as `Gates.RX`
- A list of callables such as `[Gates.RX, Gates.RY]`

See page [*Ansaetze*](ansaetze.md) for more details regarding the `Gates` class.
If a list of encodings is provided, the input is assumed to be multi-dimensional.
Otherwise multiple inputs are treated as batches of inputs.
If you want to visualize zero-valued encoding gates in the model, set `remove_zero_encoding` to `False` on instantiation.

In case of a multi-dimensional input, you can obtain the highest frequency in each encoding dimension from the `model.frequencies` property.
Now, `model.degree` in turn will reflect the highest number in this list.

## State Preparation

While the encoding is applied in each data-reuploading step, the state preparation is only applied at the beginning of the circuit, but after the `StatePreparation` noise (see [below](#noise) for details).
The default is no state preparation. Similar to the encoding, you can provide the `state_preparation` argument as

- A string such as `"H"` that will result in a single Hadamard per qubit
- A list of strings such as `["H", "H"]` that will result in two consecutive Hadamards per qubit
- Any callable such as `Gates.H`
- A list of callables such as `[Gates.H, Gates.H]`

See page [*Ansaetze*](ansaetze.md) for more details regarding the `Gates` class.

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
- `MultiQubitDepolarizing`
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

## Multiprocessing

Our framework can parallelise the execution of the model by providing a `mp_threshold` parameter (defaults to -1).
This parameter effectively determines the batch size above which the model is executed in parallel.
Given a parameter shape of, i.e. `[x,y,1000]` and a `mp_threshold` of 400, three separate processes will be launched.
If there are only two processes available on the machine, then the model will execute only two processes concurrently, wait for them to finish and then execute the remaining process.

```
n_samples = 4500

model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type="Circuit_19",
    mp_threshold=1000,
)
```

Depending on the chosen parameters and your machine, this can result in a significant speedup.
Note however, that this is currently only available for `n_qubits<model.lightning_threshold` which is 12 by default.
Above this threshold, Pennylane's `lightning.qubit` device is used which would interfere with an additional parallelism.
Also note, that no checks on the available memory will be performed and that the memory consumption could multiply with the number of parallel processes.

Multiprocessing works for both parameters and inputs, meaning that if a batched input is provided, processing will be parallelized in the same way as explained above.
Note, that if both, parameters and inputs are batched with size `B_I` and `B_P` respectively, the effective batch dimension will multiply, i.e. resulting in `B_I * B_P` combinations. 
Internally, these combinations will be flattened during processing and then reshaped to the original shape afterwards, such that the output shape is `[O, B_I, B_P]`.
Here, `O` is the general output shape depending on the execution type, `B_I` is the batch dimension of the inputs and `B_P` is the batch dimension of the parameters.
This shape is also available as a property of the model: `model.batch_shape`.

Naturally, the question arises which is the best choice for the hyperparameter `mp_threshold` as a higher value will result in fewer processes being spawned, while a lower value might over-allocate the CPU and adds parallelization overhead which reduces the speedup compared to single process.
To visualize this, we provide following Figure where we computed the speedup for several different configurations of `mp_threshold` and `n_samples` with a 4 qubit circuit, averaging over 8 runs.

![Multiprocessing](figures/mp_result_density_light.png#only-light)
![Multiprocessing](figures/mp_result_density_dark.png#only-dark)

The computation was performed on a 16 core CPU with 32GB of RAM.
It is clearly visible, that e.g. a `mp_threshold` of 500 saturates the multi-processing capability after 4500 samples similar to a `mp_threshold` of 1000 at 9000 samples.
Also note how the speedup (over single process) is 1 until the number of samples equal `mp_threshold`.

## Quantikz Export

In addition to the printing the model to console and into a figure using matplotlib (thanks to Pennylane); our framework extends this functionality by allowing you to create nice [Quantikz](https://doi.org/10.48550/arXiv.1809.03842) figures that you can embedd in a Latex document :heart_eyes:.
This can be achieved by 

```python
fig = model.draw(figure="tikz", inputs_symbols="x", gate_values=False)
fig.export("tikz_circuit.tex", full_document=True)
```

![Tikz Circuit](figures/circuit_tikz_light.png#only-light)
![Tikz Circuit](figures/circuit_tikz_dark.png#only-dark)

Inputs are represented with "x" by default, which can be changed by adjusting the optional parameter `inputs_symbols`.
If you want to see the actual gate values instead of variables, simply set `gate_values=True` which is also the default option.
The returned `fig` variable is a `TikzFigure` object that stores the Latex string and allows exporting to a specified file.
To create a document that can be compiled, simply pass `full_document=True` when calling `export`.