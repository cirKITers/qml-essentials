# expval
uv run python mp_experiment.py --seed 1000 --execution_type expval --min_n_samples 500 --max_n_samples 100000 --n_samples_step 2500 --n_qubits 4 --n_layers 1 --n_runs 1

# density
uv run python mp_experiment.py --seed 1000 --execution_type density --min_n_samples 500 --max_n_samples 30000 --n_samples_step 5000 --n_qubits 4 --n_layers 1 --n_runs 1
