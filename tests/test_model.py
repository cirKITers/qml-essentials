from qml_essentials.model import Model


def test_parameters() -> None:
    test_cases = [
        {
            "shots": -1,
            "state_vector": False,
            "exp_val": True,
            "exception": False,
        },
        {
            "shots": -1,
            "state_vector": True,
            "exp_val": False,
            "exception": False,
        },
        {
            "shots": 1024,
            "state_vector": False,
            "exp_val": False,
            "exception": False,
        },
        {
            "shots": 1024,
            "state_vector": False,
            "exp_val": True,
            "exception": False,
        },
        {
            "shots": -1,
            "state_vector": True,
            "exp_val": True,
            "exception": True,
        },
        {
            "shots": 1024,
            "state_vector": True,
            "exp_val": False,
            "exception": True,
        },
    ]

    for test_case in test_cases:
        model = Model(
            n_qubits=2,
            n_layers=1,
            circuit_type="Circuit_19",
            data_reupload=True,
            initialization="random",
            output_qubit=0,
            shots=test_case["shots"],
        )

        try:
            result = model(
                model.params,
                inputs=None,
                noise_params=None,
                cache=False,
                state_vector=test_case["state_vector"],
                exp_val=test_case["exp_val"],
            )
            print(f"Test case {test_case}: {result}")
        except Exception as e:
            assert test_case[
                "exception"
            ], f"Got exception with configuration {test_case}: {e}"
            print(f"Exception as intended for {test_case}: {e}")
