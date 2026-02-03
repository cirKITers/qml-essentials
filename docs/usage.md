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

![Hardware Efficient Ansatz](figures/hae_light.png#center#only-light)
![Hardware Efficient Ansatz](figures/hae_dark.png#center#only-dark)

Looks good to you? :eyes: Head over to the [*Training*](training.md) page for **getting started** with an easy example, where we also show how to implement **trainable frequencies** :rocket:
If you want to learn more about, why we get the above results, checkout the [*Data-Reuploading*](#data-reuploading) section.

Note that calling the model without any (`None`) values for the `params` and `inputs` argument, will implicitly call the model with the recently (or initial) parameters and `0`s as input.
I.e. simply running the following
```python
model()
```
will return the combined expectation value of a n-local measurement (`output_qubit=-1` is default). 

In the following we will describe some concepts of the `Model` class.
For a more detailled reference on the methods and arguments that are available, please see the [references page](https://cirkiters.github.io/qml-essentials/references/#model).

## The essentials

There is much more to this package than just providing a Fourier model.
You can calculate the [Expressibility](expressibility.md) or [Entangling Capability](entanglement.md) besides the [Coefficients](coefficients.md) which are unique to this kind of QML interpretation.
You can also provide a custom circuit, by instantiating from the `Circuit` class in `qml_essentials.ansaetze.Circuit`.
See page [*Ansaetze*](ansaetze.md) for more details and a list of available Ansatzes that we provide with this package.

## Data-Reuploading

The idea of repeating the input encoding is one of the core features of our framework and builds upon the work by [*Schuld et al. (2020)*](https://doi.org/10.48550/arXiv.2008.08605).
Essentially, it allows us to represent a quantum circuit as a truncated Fourier series, which is a powerful feature that enables the model to mimic arbitrary non-linear functions.
The number of frequencies that the model can represent is constrained by the number of data encoding steps within the circuit.

Typically, there is a reuploading step after each layer and on each qubit (`data_reupload=True`).
However, our package also allows you to specify an array with the number of rows representing the qubits and number of columns representing the layers.
Then, a `True` means that encoding is applied at the corresponding position within the circuit.

In the following example, we disable two instances of the data-reuploading step, thus leaving the model with `model.degree = (5)` frequencies (2 negative + zero frequency + 2 positive).

```python
model = Model(
    n_qubits=2,
    n_layers=2,
    circuit_type="Hardware_Efficient",
    data_reupload=[[True, False], [False, True]],
)
```

Checkout the [*Coefficients*](coefficients.md) page for more details on how you can visualize such a model using tools from signal analysis.
If you want to encode multi-dimensional data (check out the [*Encoding*](usage.md#encoding) section on how to do that), you can specify another dimension in the `data_reupload` argument (which just extents naturally).
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
Given a PRNG key, it returns the result of `random.split(key)` (i.e. a new key and its subkey) as documented [here](https://docs.jax.dev/en/latest/random-numbers.html).
This allows to repetively call `key, _ = model.initialize_params(key)` to generate a continous sequence of random initializations.

## Encoding

The encoding can be set when instantiating the model with the `encoding` argument.

The default encoding is "RX" which will result in a single RX rotation per qubit.
Other options are:

- A string such as `"RX"` that will result in a single RX rotation per qubit
- A list of strings such as `["RX", "RY"]` that will result in a sequential RX and RY rotation per qubit
- Any callable such as `Gates.RX`
- A list of callables such as `[Gates.RX, Gates.RY]`
- An instance of the `Encoding` class

See page [*Ansaetze*](ansaetze.md) for more details regarding the `Gates` class.
If a list of encodings is provided, the input is assumed to be multi-dimensional.
Otherwise multiple inputs are treated as batches of inputs.
If you want to visualize zero-valued encoding gates in the model, set `remove_zero_encoding` to `False` on instantiation.

In case of a multi-dimensional input, you can obtain the highest frequency in each encoding dimension from the `model.degree` property.
Note that, `model.degree` includes the negative and zero frequency (i.e. the full spectrum).
Individual frequencies can be obtained via `model.frequencies`.

By default, all encodings are `Hamming` encodings, meaning, all encodings are applied equally in each data-reuploading step.
Note it is also possible to provide a custom encoding as the `encoding` argument essentially accepts any callable or list of callables see [here](ansaetze.md#custom-encoding) for more details.
To make things a little bit easier, we implement following encoding strategies as introduced in [Generalization despite overfitting in quantum machine learning models](https://doi.org/10.22331/q-2023-12-20-1210) with their respective spectral properties:

| Encoding strategy | Spectrum $\Omega$                                                                                                  | $\vert\Omega\vert$ |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------ |
| Hamming           | $\{-n_{q},-(n_{q}-1),\ldots,n_{q}\}$                                                                               | $2 n_{q}+1$        |
| Binary            | $\{-2^{n_{q}}+1,\ldots,2^{n_{q}}-1\}$                                                                              | $2^{n_{q}+1}- 1$   |
| Ternary           | $\left\{-\left\lfloor\frac{3^{n_{q}}}{2}\right\rfloor,\ldots,\left\lfloor\frac{3^{n_{q}}}{2}\right\rfloor\right\}$ | $3^{n_{q}}$        |

You can use these templates by instantiating an `Encoding` class with the encoding strategy you like and passing it to the model upon initialization:

```python
from qml_essentials.ansaetze import Encoding

model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type="Circuit_19",
    encoding=Encoding("ternary", ["RX", "RY"]),
)

model.frequencies
```

Returns `[9,9]`, which corresponds to the ternary spectrum $3^{2}$ for two indpendent inputs.


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
Setting `output_qubit` to an integer will measue the qubit with the index specified.
Furthermore, "parity measurements" are supported, where `output_qubit` becomes a list of qubit pairs, e.g. `[[0, 1], [2, 3]]` to measure the parity between qubits 0 and 1 and qubits 2 and 3.

If `force_mean` flag is set when calling the model, the output is averaged to a single value (while keeping the batch/ input dimension).
This is usually helpful, if you want to perform a n-local measurement over all qubits where only the average over $n$ expecation values is of interest.

## Execution Type

Our model be simulated in different ways by setting the `execution_type` property, when calling the model, to:

- `expval`: Returns the expectation value between $0$ and $1$
- `density`: Calculates the density matrix
- `probs`: Simulates the model with the number of shots, set by `model.shots`

For all three different execution types, the output shape is determined by the `output_qubit` argument, provided in the instantiation of the model.
In case of `density` the partial density matrix is returned.

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

## Pulse Level Simulation

Our framework extends beyond unitary-level simulation by integrating **pulse-level simulation** through [PennyLane’s pulse module](https://docs.pennylane.ai/en/stable/code/qml_pulse.html).  
This allows you to move from the abstract unitary layer, where gates are treated as instantaneous idealized operations, down to the physical pulse layer, where gates are represented by time-dependent microwave control fields.  

In the pulse representation, each gate is decomposed into Gaussian-shaped pulses parameterized by:

- $A$: amplitude of the pulse
- $\sigma$: width (standard deviation) of the Gaussian envelope
- $t$: pulse duration

By default, the framework provides optimized pulse parameters based on typical superconducting qubit frequencies ($\omega_q = 10\pi$, $\omega_c = 10\pi$).  

Switching between unitary-level and pulse-level execution is seamless and controlled via the `gate_mode` argument:

```python
# Default unitary-level simulation
model(params, inputs)

# Pulse-level simulation
model(params, inputs, gate_mode="pulse")
```

Pulse-level gates can also be instantiated directly:

```python
from qml_essentials.ansaetze import Gates

# RX gate represented by its microwave pulse
Gates.RX(w, wires=0, gate_mode="pulse")

# With custom pulse parameters [A, sigma, t]
pulse_params = [0.5, 0.2, 1.0]
Gates.RX(w, wires=0, pulse_params=pulse_params, gate_mode="pulse")
```
and then used in [custom Ansaetze](ansaetze.md#custom_ansatz) or directly as [encoding gates](ansaetze.md#custom_encoding).
See our documentation on [Quantum Optimal Control (QOC)](ansaetze.md#quantum_optimal_control_qoc) for more details on how to choose pulse parameters.

For more details:

- See [*Ansaetze*](ansaetze.md#pulse_simulation) for a deeper explanation of our pulse-level gates and ansaetze, as well as details on Quantum Optimal Control (QOC), which enables optimizing pulses directly for target unitaries.  
- See [*Training*](training.md#pulse_level) for how to train pulse parameters jointly with rotation angles.  


## Multithreading (using JAX)

Our framework can parallelise the execution of the model setting the `use_multithreading` flag (defaults to False).
In our framework, JAX then automatically handles the number and distribution of the workers depending on the batch sizes and available CPUs.
```
model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type="Circuit_19",
    use_multithreading=True,
)
```

Depending your machine, this can result in a significant speedup.
Note however, that this is currently only available for `n_qubits<model.lightning_threshold` which is 12 by default.
Above this threshold, Pennylane's `lightning.qubit` device is used which would interfere with an additional parallelism.

Mutlithreading works for both parameters and inputs, meaning that if a batched input is provided, processing will be parallelized in the same way as explained above.
Note, that if both, parameters and inputs are batched with size `B_I` and `B_P` respectively, the effective batch dimension will multiply, i.e. resulting in `B_I * B_P` combinations. 
Internally, these combinations will be flattened during processing and then reshaped to the original shape afterwards, such that the output shape is `[O, B_I, B_P]`.
Here, `O` is the general output shape depending on the execution type, `B_I` is the batch dimension of the inputs and `B_P` is the batch dimension of the parameters.
This shape is also available as a property of the model: `model.batch_shape`.
Note, that the output shape is always squeezed, i.e. batch axes will be suppressed if their dimension is 1.
Also, there is a third batch axis in `model.batch_shape` for pulse parameters.
See more on that topic in [*Ansaetze*](ansaetze.md#pulse_simulation).

In addition to letting the model handle repeating the batch axes, it is also possible to disable this functionality by setting `repeat_batch_axis` upon model initialization.
This parameter is an array of boolean values determining of the corresponding axis in the `batch_shape` (#Inputs, #Params, #PulseParams)should be repeated.
Of course, when providing the batch manually, the dimensions have to match.

```python
model = Model(
    n_qubits=2,
    n_layers=1,
    circuit_type="Circuit_19",
    repeat_batch_axis=[False, True, True],
    use_multithreading=True,
)

key = jax.random.key(1000)
key = model.initialize_params(key, repeat=10)
model(inputs=random.uniform(key, (10, 1)))
```
In this example, instead of a batch size of `100`, the output will have a batch size of `10` instead (shape `(10,2)`).


For density matrix calculations, we computed the speedup of a multi-threaded computation over a single-threaded computation with a 4 qubit circuit, averaged over 8 runs, as shown in the following figure.

![Multiprocessing Density](figures/mp_result_density_light.png#center#only-light)
![Multiprocessing Density](figures/mp_result_density_dark.png#center#only-dark)

The computation was performed on a 16 core CPU with 32GB of RAM.

While computing the expectation value is significantly easier, there can still be a speedup achieved, as shown in the following figure.

![Multiprocessing Expval](figures/mp_result_expval_light.png#center#only-light)
![Multiprocessing Expval](figures/mp_result_expval_dark.png#center#only-dark)


## Quantikz Export

In addition to the printing the model to console and into a figure using matplotlib (thanks to Pennylane); our framework extends this functionality by allowing you to create nice [Quantikz](https://doi.org/10.48550/arXiv.1809.03842) figures that you can embedd in a Latex document :heart_eyes:.
This can be achieved by 

```python
fig = model.draw(figure="tikz", inputs_symbols="x", gate_values=False)
fig.export("tikz_circuit.tex", full_document=True)
```

![Tikz Circuit](figures/circuit_tikz_light.png#center#only-light)
![Tikz Circuit](figures/circuit_tikz_dark.png#center#only-dark)

Inputs are represented with "x" by default, which can be changed by adjusting the optional parameter `inputs_symbols`.
If you want to see the actual gate values instead of variables, simply set `gate_values=True` which is also the default option.
The returned `fig` variable is a `TikzFigure` object that stores the Latex string and allows exporting to a specified file.
To create a document that can be compiled, simply pass `full_document=True` when calling `export`.
