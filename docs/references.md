## Ansaetze

```python
from qml_essentials.ansaetze import Ansaetze
```

::: qml_essentials.ansaetze.Ansaetze
    options:
      heading_level: 3

### Circuit

```python
from qml_essentials.ansaetze import Circuit
```

::: qml_essentials.ansaetze.Circuit
    options:
      heading_level: 4

### Declarative Circuit

```python
from qml_essentials.ansaetze import DeclarativeCircuit
```

::: qml_essentials.ansaetze.DeclarativeCircuit
    options:
      heading_level: 4

### Block

```python
from qml_essentials.ansaetze import Block
```

::: qml_essentials.ansaetze.Block
    options:
      heading_level: 4

### Encoding

```python
from qml_essentials.ansaetze import Encoding
```

::: qml_essentials.ansaetze.Encoding
    options:
      heading_level: 4

## Gates

As the structure of the different classes used to realize pulse and unitary gates can be a bit confusing, the following diagram might help:

![Gate Structure](figures/pulses_structure_light.png#center#only-light)
![Gate Structure](figures/pulses_structure_dark.png#center#only-dark)

```python
from qml_essentials.gates import Gates
```

::: qml_essentials.gates.Gates
    options:
      heading_level: 3

### Unitary Gates

```python
from qml_essentials.gates import UnitaryGates
```

::: qml_essentials.gates.UnitaryGates
    options:
      heading_level: 4

### Pulse Gates

```python
from qml_essentials.gates import PulseGates
```

::: qml_essentials.gates.PulseGates
    options:
      heading_level: 4

### Pulse Structure

```python
from qml_essentials.gates import PulseParams
```

::: qml_essentials.gates.PulseParams
    options:
      heading_level: 4

### Pulse Envelope

```python
from qml_essentials.gates import PulseEnvelope
```

::: qml_essentials.gates.PulseEnvelope
    options:
      heading_level: 4

### Pulse Information

```python
from qml_essentials.gates import PulseInformation
```

::: qml_essentials.gates.PulseInformation
    options:
      heading_level: 4

## Model

```python
from qml_essentials.model import Model
```

::: qml_essentials.model.Model
    options:
      heading_level: 3

## Entanglement

```python
from qml_essentials.entanglement import Entanglement
```

::: qml_essentials.entanglement.Entanglement
    options:
      heading_level: 3

## Expressibility

```python
from qml_essentials.expressibility import Expressibility
```

::: qml_essentials.expressibility.Expressibility
    options:
      heading_level: 3

## Coefficients

```python
from qml_essentials.coefficients import Coefficients
```

::: qml_essentials.coefficients.Coefficients
    options:
      heading_level: 3

### Fourier Tree

```python
from qml_essentials.coefficients import FourierTree
```

::: qml_essentials.coefficients.FourierTree
    options:
      heading_level: 4

### Fourier Coefficient Correlation

```python
from qml_essentials.coefficients import FCC
```

::: qml_essentials.coefficients.FCC
    options:
      heading_level: 4

### Datasets

```python
from qml_essentials.coefficients import Datasets
```

::: qml_essentials.coefficients.Datasets
    options:
      heading_level: 4

## Topologies

```python
from qml_essentials.topologies import Topology
```

::: qml_essentials.topologies.Topology
    options:
      heading_level: 3

## Operations

```python
from qml_essentials.operations import Operation
```

::: qml_essentials.operations.Operation
    options:
      heading_level: 3

### Hermitian

```python
from qml_essentials.operations import Hermitian
```

::: qml_essentials.operations.Hermitian
    options:
      heading_level: 4

### Kraus Channel

```python
from qml_essentials.operations import KrausChannel
```

::: qml_essentials.operations.KrausChannel
    options:
      heading_level: 4

## Math

```python
from qml_essentials.math import fidelity, trace_distance, phase_difference
```

::: qml_essentials.math.fidelity
    options:
      heading_level: 3

::: qml_essentials.math.trace_distance
    options:
      heading_level: 3

::: qml_essentials.math.phase_difference
    options:
      heading_level: 3

## Quantum Optimal Control

```python
from qml_essentials.qoc import QOC
```

::: qml_essentials.qoc.QOC
    options:
      heading_level: 3

### Cost Functions

```python
from qml_essentials.qoc import Cost
```

::: qml_essentials.qoc.Cost
    options:
      heading_level: 4

### Cost Function Registry

```python
from qml_essentials.qoc import CostFnRegistry
```

::: qml_essentials.qoc.CostFnRegistry
    options:
      heading_level: 4

## Yaqsi

```python
from qml_essentials.yaqsi import Script
```

::: qml_essentials.yaqsi.Script
    options:
      heading_level: 3

### Yaqsi Core

```python
from qml_essentials.yaqsi import Yaqsi
```

::: qml_essentials.yaqsi.Yaqsi
    options:
      heading_level: 4

## Drawing

```python
from qml_essentials.drawing import TikzFigure
```

::: qml_essentials.drawing.TikzFigure
    options:
      heading_level: 3

```python
from qml_essentials.drawing import PulseEvent
```

::: qml_essentials.drawing.PulseEvent
    options:
      heading_level: 3

## Tape

```python
from qml_essentials.tape import recording, pulse_recording
```

::: qml_essentials.tape.recording
    options:
      heading_level: 3

::: qml_essentials.tape.pulse_recording
    options:
      heading_level: 3