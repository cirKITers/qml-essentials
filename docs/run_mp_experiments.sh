# expval
poetry run python mp_experiment.py --seed 1000 --execution_type expval --min_n_samples 3000 --max_n_samples 60000 --n_samples_step 3000 --n_qubits 4 --min_mp_threshold 1000 --max_mp_threshold 10000 --mp_threshold_step 1000 --n_layers 1 --n_runs 1

# density
poetry run python mp_experiment.py --seed 1000 --execution_type density --min_n_samples 500 --max_n_samples 10000 --n_samples_step 500 --n_qubits 4 --min_mp_threshold 500 --max_mp_threshold 5000 --mp_threshold_step 500 --n_layers 1 --n_runs 1
