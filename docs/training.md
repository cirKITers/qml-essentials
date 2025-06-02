# Training

This section describes how to use the model provided with this package, using a simple training scenario as an example.

We consider a Fourier series with $n$ frequencies defined as follows:

\[
f(x, \boldsymbol{\theta})=\sum_{\omega \in \boldsymbol{\Omega}} c_{\omega}(\boldsymbol{\theta}) e^{i \omega x}=\sum_{\omega \in \boldsymbol{\Omega}} c_{\omega}(\boldsymbol{\theta}) \left(\cos(\omega x) + i \sin(\omega x)\right)
\]

Here, $\omega \in \boldsymbol{\Omega}$ are the frequencies in the spectrum with the Fourier coefficients $c_{\omega}(\boldsymbol{\theta})$, parameterized by the set of trainable parameters $\boldsymbol{\theta}$.

As shown by [Schuld et al. (2020)](https://arxiv.org/abs/2008.08605), a quantum circuit, parametrised by $\boldsymbol{\theta}$ and input $x$ and is equivalent to the Fourier series representation.
Such circuits must be of the following form:

\[
f(x, \boldsymbol{\theta})=\langle 0\vert^{\otimes n} U^{\dagger}(x, \boldsymbol{\theta}) \mathcal{M} U(x, \boldsymbol{\theta})\vert 0\rangle^{\otimes n}
\]

Therefore, training such a model on a Fourier series is a proof-of-concept which we want to demonstrate here.

Let's start with building our dataset. A Fourier series with $4$ frequencies:
```python
import pennylane.numpy as np
import matplotlib.pyplot as plt

domain = [-np.pi, np.pi]
omegas = np.array([1, 2, 3, 4])
coefficients = np.array([0.5, 0.5, 0.5, 0.5])

# Calculate the number of required samples to satisfy the Nyquist criterium
n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))
# Sample the domain linearly
x = np.linspace(domain[0], domain[1], num=n_d)

# define our Fourier series f(x)
def f(x):
    return 1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))

# evaluate f(x) on the domain samples
y = np.stack([f(sample) for sample in x])

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

![Fourier Series](figures/fourier_series_light.png#center#only-light)
![Fourier Series](figures/fourier_series_dark.png#center#only-dark)

Note that we chose the coefficients to be all $0.5$. Play around with those values to change the magnitude of each frequency component.
Also note that we're using the Pennylane version of Numpy, which is required because of the optimizer that we will be using later.
Now that we have our "dataset", let's move on and build a model:
```python
from qml_essentials.model import Model

model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Circuit_19",
)
```

This is the minimal amout of information needed. According to the work referenced above, a model with $4$ qubits should be capable of learning a Fourier series with $4$ frequencies, considering single qubit Pauli encoding (which we have by default).

Now, let's train our model:
```python
import pennylane as qml

opt = qml.AdamOptimizer(stepsize=0.01)

def cost_fct(params):
    y_hat = model(params=params, inputs=x, force_mean=True)

    return np.mean((y_hat - y) ** 2)

for epoch in range(1, 1001):
    model.params, cost_val = opt.step_and_cost(cost_fct, model.params)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Cost: {cost_val:.4f}")

plt.plot(x, y, label="True function")
plt.plot(x, model(params=model.params, inputs=x, force_mean=True), label="Model prediction")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
```

```
Epoch: 100, Cost: 0.0081
Epoch: 200, Cost: 0.0073
Epoch: 300, Cost: 0.0051
Epoch: 400, Cost: 0.0043
Epoch: 500, Cost: 0.0036
Epoch: 600, Cost: 0.0022
Epoch: 700, Cost: 0.0014
Epoch: 800, Cost: 0.0008
Epoch: 900, Cost: 0.0006
Epoch: 1000, Cost: 0.0001
```

![Ground Truth and Prediction](figures/trained_series_light.png#center#only-light)
![Ground Truth and Prediction](figures/trained_series_dark.png#center#only-dark)

As you can see, the model is able to learn the Fourier series with the $4$ frequencies.

For the model we just trained, we considered the best possible scenario: evenly spaced, integer omegas. But, as shown by [Schuld et al. (2020)](https://arxiv.org/abs/2008.08605), we'll need an increasing and inefficient amount of qubits for larger omegas. What is more, the model will fail altogether if the frequencies are un-evenly spaced. Luckily, [Shaderberg et al. (2024)](https://arxiv.org/abs/2309.03279) showed how we can let the model choose its own frequencies by including a set of encoding parameters that act on the input before the encoding layers of the circuit. We demonstrate this functionality below. 

Let's first slighly modify the omegas from the first example and re-generate the data:
```python
domain = [-np.pi, np.pi]
omegas = np.array([1.2, 2.6, 3.4, 4.9])
coefficients = np.array([0.5, 0.5, 0.5, 0.5])

# Calculate the number of required samples to satisfy the Nyquist criterium
n_d = int(np.ceil(2 * np.max(np.abs(domain)) * np.max(omegas)))
# Sample the domain linearly
x = np.linspace(domain[0], domain[1], num=n_d)

# define our Fourier series f(x)
def f(x):
    return 1 / np.linalg.norm(omegas) * np.sum(coefficients * np.cos(omegas.T * x))

# evaluate f(x) on the domain samples
y = np.stack([f(sample) for sample in x])

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
```

![Fourier Series](figures/fourier_series_2_light.png#center#only-light)

Now, let's build a model with fixed frequencies, as before, and one with trainable frequencies:
```python
model = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Circuit_19",
    trainable_frequencies=False
)
model_tf = Model(
    n_qubits=4,
    n_layers=1,
    circuit_type="Circuit_19",
    trainable_frequencies=True
)
```

Let's train both models:
```python
# - Fixed Frequencies -
opt = qml.AdamOptimizer(stepsize=0.01)

print("Training fixed frequency model")
for epoch in range(1, 1001):
    model.params, cost_val = opt.step_and_cost(cost_fct, model.params)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Cost: {cost_val:.4f}")

# - Trainable Frequencies -
opt = qml.AdamOptimizer(stepsize=0.01)

def cost_fct_tf(params, enc_params):
    y_hat = model_tf(params=params, enc_params=enc_params, inputs=x, force_mean=True)
    return np.mean((y_hat - y) ** 2)

print(f"\nTraining trainable frequency model")
for epoch in range(1, 1001):
    (model_tf.params, model_tf.enc_params), cost_val_tf = opt.step_and_cost(cost_fct_tf, model_tf.params, model_tf.enc_params)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Cost: {cost_val_tf:.6f}")

plt.plot(x, y, label="True function")
plt.plot(x, model(params=model.params, inputs=x, force_mean=True), label="Fixed frequencies model prediction")
plt.plot(x, model_tf(params=model_tf.params, enc_params=model_tf.enc_params, inputs=x, force_mean=True), label="Trainable frequencies model prediction")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
```

```
Training fixed frequency model
Epoch: 100, Cost: 0.0082
Epoch: 200, Cost: 0.0067
Epoch: 300, Cost: 0.0038
Epoch: 400, Cost: 0.0031
Epoch: 500, Cost: 0.0027
Epoch: 600, Cost: 0.0026
Epoch: 700, Cost: 0.0025
Epoch: 800, Cost: 0.0024
Epoch: 900, Cost: 0.0023
Epoch: 1000, Cost: 0.0023

Training trainable frequency model
Epoch: 100, Cost: 0.008454
Epoch: 200, Cost: 0.002759
Epoch: 300, Cost: 0.002382
Epoch: 400, Cost: 0.001655
Epoch: 500, Cost: 0.000232
Epoch: 600, Cost: 0.000019
Epoch: 700, Cost: 0.000010
Epoch: 800, Cost: 0.000003
Epoch: 900, Cost: 0.000001
Epoch: 1000, Cost: 0.000001
```

![Ground Truth and Prediction](figures/trained_series_2_light.png#center#only-light)

As you can see, the fixed frequencies model wasn't able to find the underlying function representing the data, while the trainable frequencies model was successful in its training.

Let's quickly check the final encoding parameter of both models:
```python
print(f"Encoding parameters of the fixed frequencies model: {model.enc_params}")
print(f"Encoding parameters of the trainable frequencies model: {np.round(model_tf.enc_params, 3)}")
```

```
Encoding parameters of the fixed frequencies model: [1. 1. 1. 1.]
Encoding parameters of the trainable frequencies model: [1.001 2.065 2.817 0.364]
```

Clearly, the trainable frequencies model found the set of encoding parameters that allowed it to represent the given arbitrary frequency spectrum. 

One last thing that might be interesting! Currently, the model applies 
```python
enc_params[qubit] * inputs[:, idx]
```
to allow for trainable frequencies. You may try different input transformations before the encoding by modifying the `model.transform_input` method. For example, if an `RX` gate performs the encoding, you may apply the identity operator by 
```python
model.transform_input = lambda inputs, qubit, idx, enc_params: np.arccos(inputs[:, idx])
```


Btw, if you're in a hurry, we have a Jupyter notebook with the exact same examples [here](https://github.com/cirKITers/qml-essentials/blob/main/docs/training.ipynb) :upside_down_face:.

Wondering what to do next? You can try a few different models, and see how they perform. If you're curious, checkout how this correlates with the [*Entanglement*](entanglement.md) and [*Expressibility*](expressibility.md) of the model.
