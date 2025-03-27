import os

from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze

ansaetze = Ansaetze.get_available()

overview_txt = ""
for ansatz in ansaetze:
    model = Model(
        n_qubits=4,
        n_layers=1,
        circuit_type=ansatz.__name__,
        output_qubit=-1,
        remove_zero_encoding=False,
    )

    fig, _ = model.draw(figure="mpl")

    cwd = os.path.dirname(__file__)
    fig.savefig(f"{cwd}/figures/{ansatz.__name__}_light.png", dpi=300)

    overview_txt += (
        f"![{ansatz.__name__}](figures/{ansatz.__name__}_light.png#only-light)\n"
    )
    overview_txt += (
        f"![{ansatz.__name__}](figures/{ansatz.__name__}_dark.png#only-dark)\n"
    )

with open("docs/ansaetze.md", "a") as f:
    f.write(overview_txt)
