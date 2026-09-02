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

to install our package from [PyPI](https://pypi.org/project/qml-essentials/).

<p align="center">
<img src="docs/figures/code.svg" width="640" title="Code Example">
</p>
<p align="center">
<img src="docs/figures/histogram.svg" width="640" title="Histogram Figure">
</p>

You can find details on how to use it and further documentation on the corresponding [Github Page](https://cirkiters.github.io/qml-essentials/).

## 📦 Package Structure

The following diagram provides an overview on how the different components within this package depend on each other.

Circuit simulation is provided by [jaqsi](https://github.com/cirKITers/jaqsi), our gate- and pulse-level simulator, which this package builds upon.

```mermaid
flowchart LR
    qmless([QML Essentials])
    qmless --> qmless.ansaetze([Ansaetze])
    qmless.ansaetze --> qmless.blocks([Blocks])
    qmless.blocks --> qmless.topo([Topologies])
    qmless.blocks --> jaqsi.gates([Gates])
    qmless --> qmless.coefficients([Coefficients])
    qmless.coefficients --> qmless.analytical([Analytical])
    qmless.coefficients --> qmless.numerical([Numerical])
    qmless.numerical --> qmless.fcc([Fourier Coefficient Correlation])
    qmless.numerical --> qmless.fingerprint([Fourier Fingerprints])
    qmless --> qmless.models([Models])
    qmless.models --> jaqsi.script([Script])
    qmless --> qmless.expr([Expressibility])
    qmless --> qmless.ent([Entanglement])
    qmless.ent --> qmless.mw([Meyer Wallach])
    qmless.ent --> qmless.re([Relative Entropy])
    qmless.ent --> qmless.bell([Bell Measurement])
    qmless.ent --> qmless.ef([Entanglement of Formation])
    qmless.ent --> qmless.ce([Concentratable Entanglement])

    subgraph jaqsi [jaqsi]
        jaqsi.gates
        jaqsi.script
    end

    classDef l1 fill:#1f8f5a,stroke:#1f8f5a,color:#d4f7e8
    classDef l2 fill:#2fb170,stroke:#2fb170,color:#d4f7e8
    classDef l3 fill:#58e3a6,stroke:#58e3a6,color:#272a35
    classDef l4 fill:#a8f0d1,stroke:#a8f0d1,color:#272a35

    linkStyle default stroke-width:2px

    class qmless l1
    class qmless.ansaetze,qmless.coefficients,qmless.models,qmless.expr,qmless.ent l2
    class qmless.blocks,qmless.numerical,qmless.analytical,qmless.mw,qmless.bell,qmless.ef,qmless.re,qmless.ce l3
    class qmless.topo,qmless.fcc,qmless.fingerprint,jaqsi.gates,jaqsi.script l4
```

## 🚧 Contributing

Contributions are highly welcome! 🤗 Take a look at our [Contribution Guidelines](https://github.com/cirKITers/qml-essentials/blob/main/CONTRIBUTING.md).

See our [coverage report](coverage/index.html) if you would like to contribute with further tests.
