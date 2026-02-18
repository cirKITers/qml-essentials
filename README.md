# QML Essentials

<p align="center">
<img src="https://raw.githubusercontent.com/cirKITers/qml-essentials/refs/heads/main/docs/logo.svg" width="200" title="Logo">
</p>
<h3 align="center">A toolbox to make working with QML models easier.</h3>
<br/>

## 📜 About

This repo contains some of the commonly used Ansaetze and coding stuff required for working with QML and Data-Reuploading models.\
There are also dedicated classes to calculate entanglement and expressiblity of a provided model as well as its Fourier coefficients.
Checkout our [Arxiv Paper](https://arxiv.org/abs/2506.06695) to learn more.

## 🚀 Getting Started

```
pip install qml-essentials
```
or with the [uv package manager](https://github.com/astral-sh/uv):
```
uv add qml-essentials
```

to install our package from [PyPI](https://pypi.org/project/qml-essentials/).

<p align="center">
<img src="docs/figures/code.svg" width="640" title="Code Example">
</p>
<p align="center">
<img src="docs/figures/histogram.svg" width="640" title="Histogram Figure">
</p>

You can find details on how to use it and further documentation on the corresponding [Github Page](https://cirkiters.github.io/qml-essentials/).

## 📦 Structure

The following diagram provides an overview on how the different components within this package depend on each other.

```mermaid
flowchart LR
    qmless[QML Essentials]
    qmless --> qmless.ansaetze[Ansaetze]
    qmless.ansaetze[Ansaetze] --> qmless.blocks[Blocks]
    qmless.blocks[Blocks] --> qmless.gates[Gates]
    qmless.blocks[Blocks] --> qmless.topo[Topologies]
    qmless.gates[Gates] --> qmless.unitary[UnitaryGates]
    qmless.gates[Gates] --> qmless.pulse[PulseGates]
    qmless --> qmless.coefficients[Coefficients]
    qmless.coefficients[Coefficients] --> qmless.analytical[Analytical]
    qmless.coefficients[Coefficients] --> qmless.numerical[Numerical]
    qmless.numerical[Numerical] --> qmless.fcc[Fourier Coefficient Correlation]
    qmless.numerical[Numerical] --> qmless.fingerprint[Fourier Fingerprints]
    qmless --> qmless.models[Models]
    qmless --> qmless.expr[Expressibility]
    qmless --> qmless.ent[Entanglement]
    qmless.ent[Entanglement] --> qmless.mw[Meyer Wallach]
    qmless.ent[Entanglement] --> qmless.ef[Entanglement of Formation]
    qmless.ent[Entanglement] --> qmless.re[Relative Entropy]
    qmless.ent[Entanglement] --> qmless.ce[Concentratable Entanglement]

    classDef l1 fill:#1f8f5a,stroke:#1f8f5a,color:#d4f7e8,rx:10,ry:10
    classDef l2 fill:#2fb170,stroke:#2fb170,color:#d4f7e8,rx:10,ry:10
    classDef l3 fill:#58e3a6,stroke:#58e3a6,color:#272a35,rx:10,ry:10
    classDef l4 fill:#a8f0d1,stroke:#a8f0d1,color:#272a35,rx:10,ry:10

    class qmless l1
    class qmless.ansaetze,qmless.coefficients,qmless.models,qmless.expr,qmless.ent l2
    class qmless.blocks,qmless.gates,qmless.numerical,qmless.analytical,qmless.mw,qmless.ef,qmless.re,qmless.ce l3
    class qmless.unitary,qmless.pulse,qmless.topo,qmless.fcc,qmless.fingerprint l4
```

## 🚧 Contributing

Contributions are highly welcome! 🤗 Take a look at our [Contribution Guidelines](https://github.com/cirKITers/qml-essentials/blob/main/CONTRIBUTING.md).

See our [coverage report](coverage/index.html) if you would like to contribute with further tests.
