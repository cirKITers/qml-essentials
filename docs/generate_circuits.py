import os

from qml_essentials.model import Model
from qml_essentials.ansaetze import Ansaetze

ansaetze = Ansaetze.get_available()

overview_txt = "\n"
for ansatz in ansaetze:
    model = Model(
        n_qubits=4,
        n_layers=1,
        circuit_type=ansatz.__name__,
        output_qubit=-1,
        remove_zero_encoding=True,
        data_reupload=False,
    )

    fig, _ = model.draw(figure="mpl")

    cwd = os.path.dirname(__file__)
    fig.savefig(
        f"{cwd}/figures/{ansatz.__name__}_light.png",
        dpi=100,
        transparent=True,
        bbox_inches="tight",
    )

    overview_txt += f"### {ansatz.__name__.replace('_', ' ')}\n"
    overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/{ansatz.__name__}_light.png#circuit#only-light)\n"
    overview_txt += f"![{ansatz.__name__.replace('_', ' ')}](figures/{ansatz.__name__}_dark.png#circuit#only-dark)\n"
    overview_txt += "\n"

with open(f"{cwd}/ansaetze.md", "a") as f:
    f.write(overview_txt)
