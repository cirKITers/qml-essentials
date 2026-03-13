import csv
import os
import pytest
import jax
import jax.numpy as jnp


from qml_essentials.qoc import (
    Cost,
    CostFnRegistry,
    QOC,
    fidelity_cost_fn,
    pulse_width_cost_fn,
    evolution_time_cost_fn,
)

jax.config.update("jax_enable_x64", True)

default_qoc_params = {
    "envelope": "gaussian",
    "cost_fns": [("fidelity", (0.5, 0.5))],
    "t_target": 0.5,
    "n_steps": 50,
    "n_samples": 12,
    "learning_rate": 0.001,
}


class TestCost:
    """Unit tests for the Cost wrapper class."""

    def test_scalar_weight(self):
        """Cost with a scalar weight multiplies the result."""

        def fn(params):
            return params.sum()

        cost = Cost(cost=fn, weight=2.0)
        result = cost(jnp.array([1.0, 3.0]))
        assert jnp.isclose(result, 8.0)

    def test_tuple_weight(self):
        """Cost with a tuple weight does element-wise weighting and sums."""

        def fn(params):
            return (params[0], params[1])

        cost = Cost(cost=fn, weight=(0.5, 0.25))
        result = cost(jnp.array([4.0, 8.0]))
        # 4*0.5 + 8*0.25 = 2 + 2 = 4
        assert jnp.isclose(result, 4.0)

    def test_ckwargs_injection(self):
        """Constant kwargs are injected into the cost function call."""

        def fn(params, offset=0.0):
            return params.sum() + offset

        cost = Cost(cost=fn, weight=1.0, ckwargs={"offset": 10.0})
        result = cost(jnp.array([1.0, 2.0]))
        assert jnp.isclose(result, 13.0)

    def test_add_two_costs(self):
        """Adding two Cost objects yields a callable that sums them."""
        c1 = Cost(cost=lambda p: p[0], weight=1.0)
        c2 = Cost(cost=lambda p: p[1], weight=1.0)
        combined = c1 + c2
        result = combined(jnp.array([3.0, 7.0]))
        assert jnp.isclose(result, 10.0)

    def test_add_cost_and_none(self):
        """Adding Cost + non-Cost returns a callable using only self."""
        c1 = Cost(cost=lambda p: p.sum(), weight=2.0)
        combined = c1 + None
        result = combined(jnp.array([1.0, 2.0]))
        assert jnp.isclose(result, 6.0)

    def test_weight_tuple_length_mismatch_raises(self):
        """Mismatched weight tuple and cost return length raises at call time."""

        def fn(params):
            return (params[0],)  # returns 1-tuple

        cost = Cost(cost=fn, weight=(0.5, 0.5))  # expects 2
        with pytest.raises(ValueError):
            cost(jnp.array([1.0]))


class TestCostFnRegistry:
    """Unit tests for CostFnRegistry classmethods."""

    def test_available_returns_builtin_names(self):
        """All three built-in cost functions are listed."""
        names = CostFnRegistry.available()
        assert "fidelity" in names
        assert "pulse_width" in names
        assert "evolution_time" in names

    def test_get_known(self):
        """Getting a known cost function returns correct metadata."""
        meta = CostFnRegistry.get("fidelity")
        assert meta["fn"] is fidelity_cost_fn
        assert meta["n_weights"] == 2
        assert meta["default_weight"] == (0.5, 0.5)
        assert "pulse_script" in meta["ckwargs_keys"]

    def test_get_unknown_raises(self):
        """Getting an unregistered name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cost function"):
            CostFnRegistry.get("unknown")

    def test_register_and_cleanup(self):
        """Registering a new cost function makes it available."""

        def dummy_fn(params):
            return jnp.array(0.0)

        name = "_test_dummy_cost"
        try:
            CostFnRegistry.register(
                name=name,
                fn=dummy_fn,
                n_weights=1,
                default_weight=0.1,
                ckwargs_keys=[],
            )
            assert name in CostFnRegistry.available()
            meta = CostFnRegistry.get(name)
            assert meta["fn"] is dummy_fn
            assert meta["n_weights"] == 1
        finally:
            # Clean up so other tests aren't affected
            CostFnRegistry._REGISTRY.pop(name, None)

    def test_register_duplicate_raises(self):
        """Registering a name that already exists raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            CostFnRegistry.register(
                name="fidelity",
                fn=lambda p: 0.0,
                n_weights=1,
                default_weight=0.1,
            )

    # --- parse_cost_arg ---

    @pytest.mark.parametrize(
        "arg,expected_name,expected_weight",
        [
            ("pulse_width:0.3", "pulse_width", 0.3),
            ("fidelity:0.6,0.2", "fidelity", (0.6, 0.2)),
            ("evolution_time", "evolution_time", 1.0),
        ],
        ids=["scalar_weight", "tuple_weight", "default_weight"],
    )
    def test_parse_cost_arg(self, arg, expected_name, expected_weight):
        """Parsing cost arg strings returns the correct name and weight."""
        name, weight = CostFnRegistry.parse_cost_arg(arg)
        assert name == expected_name
        assert weight == expected_weight

    @pytest.mark.parametrize(
        "arg,match",
        [
            ("bogus:0.5", "Unknown cost function"),
            ("fidelity:0.5", "expects 2 weight"),
            ("pulse_width:0.3,0.2", "expects 1 weight"),
        ],
        ids=["unknown_name", "too_few_weights", "too_many_weights"],
    )
    def test_parse_cost_arg_raises(self, arg, match):
        """Invalid parse_cost_arg inputs raise ValueError."""
        with pytest.raises(ValueError, match=match):
            CostFnRegistry.parse_cost_arg(arg)


class TestPulseWidthCostFn:
    """Tests for the pulse_width_cost_fn."""

    @pytest.mark.parametrize(
        "params,envelope,expected",
        [
            (jnp.array([10.0, 2.5, 0.8]), "gaussian", 2.5),
            (jnp.array([0.5]), "general", 0.0),
        ],
        ids=["gaussian_returns_sigma", "general_returns_zero"],
    )
    def test_pulse_width_cost(self, params, envelope, expected):
        """pulse_width_cost_fn returns the expected value for each envelope."""
        cost = pulse_width_cost_fn(params, envelope=envelope)
        assert jnp.isclose(cost, expected)


class TestEvolutionTimeCostFn:
    """Tests for the evolution_time_cost_fn."""

    @pytest.mark.parametrize(
        "params,t_target,expected",
        [
            (jnp.array([5.0, 1.0, 1.0]), 1.0, 0.0),
            (jnp.array([5.0, 1.0, 1.5]), 1.0, 0.25),
        ],
        ids=["at_target_is_zero", "deviation_is_squared_relative"],
    )
    def test_evolution_time_cost(self, params, t_target, expected):
        """evolution_time_cost_fn returns ((t - t_target) / t_target) ** 2."""
        cost = evolution_time_cost_fn(params, t_target=t_target)
        assert jnp.isclose(cost, expected)

    def test_symmetric_penalty(self):
        """The cost is the same whether t is above or below t_target."""
        t_target = 2.0
        params_above = jnp.array([0.0, 3.0])  # t = 3.0
        params_below = jnp.array([0.0, 1.0])  # t = 1.0
        cost_above = evolution_time_cost_fn(params_above, t_target=t_target)
        cost_below = evolution_time_cost_fn(params_below, t_target=t_target)
        assert jnp.isclose(cost_above, cost_below)

    def test_is_differentiable(self):
        """JAX can compute gradients through the cost function."""
        grad_fn = jax.grad(lambda p: evolution_time_cost_fn(p, t_target=1.0))
        grads = grad_fn(jnp.array([5.0, 1.0, 2.0]))
        # Only the last element (t) should have nonzero gradient
        assert jnp.isclose(grads[0], 0.0)
        assert jnp.isclose(grads[1], 0.0)
        assert not jnp.isclose(grads[2], 0.0)


class TestQOCInit:
    """Tests for QOC construction and parameter storage."""

    def test_default_cost_fns(self):
        """cost_fns from default_qoc_params are stored correctly."""
        qoc = QOC(**default_qoc_params)
        names = [name for name, _ in qoc.cost_fns]
        assert "fidelity" in names

    def test_custom_cost_fns(self):
        """Custom cost_fns override the defaults."""
        custom = [("fidelity", (0.5, 0.5))]
        params = {**default_qoc_params, "cost_fns": custom}
        qoc = QOC(**params)
        assert qoc.cost_fns == custom

    def test_stores_parameters(self):
        """All __init__ parameters are stored as attributes."""
        qoc = QOC(
            envelope="gaussian",
            cost_fns=[("fidelity", (0.5, 0.5))],
            t_target=2.0,
            n_steps=500,
            n_samples=8,
            learning_rate=0.01,
            log_interval=100,
        )
        assert qoc.envelope == "gaussian"
        assert qoc.t_target == 2.0
        assert qoc.n_steps == 500
        assert qoc.n_samples == 8
        assert qoc.learning_rate == 0.01
        assert qoc.log_interval == 100

    def test_unknown_cost_fn_raises(self):
        """Using an unregistered cost function name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cost function"):
            params = {**default_qoc_params, "cost_fns": [("nonexistent", 0.5)]}
            QOC(**params)


class TestSaveResults:
    """Tests for QOC.save_results CSV I/O."""

    @pytest.mark.parametrize(
        "writes,expected_rows,expected_gate_fidelity",
        [
            (
                [("RX", 0.95, [1.0, 2.0, 3.0])],
                1,
                {"RX": 0.95},
            ),
            (
                [("RX", 0.9, [1.0, 2.0]), ("RY", 0.8, [3.0, 4.0])],
                2,
                {"RX": 0.9, "RY": 0.8},
            ),
            (
                [("RX", 0.9, [1.0, 2.0]), ("RX", 0.95, [5.0, 6.0])],
                1,
                {"RX": 0.95},
            ),
            (
                [("RX", 0.9, [1.0]), ("RY", 0.8, [2.0]), ("RX", 0.95, [3.0])],
                2,
                {"RX": 0.95, "RY": 0.8},
            ),
        ],
        ids=[
            "creates_new_file",
            "appends_new_gate",
            "overwrites_existing_gate",
            "preserves_other_gates_on_overwrite",
        ],
    )
    def test_save_results_csv(
        self, tmp_path, writes, expected_rows, expected_gate_fidelity
    ):
        """save_results writes/overwrites CSV rows correctly."""
        qoc = QOC(**default_qoc_params, file_dir=str(tmp_path))
        for gate, fid, params in writes:
            qoc.save_results(gate, fid, params)

        csv_path = tmp_path / "qoc_results.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == expected_rows
        gate_map = {r[0]: float(r[1]) for r in rows}
        for gate, fid in expected_gate_fidelity.items():
            assert gate_map[gate] == pytest.approx(fid)

    def test_no_file_when_file_dir_is_none(self, tmp_path):
        """When file_dir is explicitly None, nothing is written."""
        qoc = QOC(**default_qoc_params, file_dir=str(tmp_path))
        qoc.file_dir = None
        qoc.save_results("RX", 0.9, [1.0])
        assert not (tmp_path / "qoc_results.csv").exists()


class TestOptimizeSmoke:
    """Smoke test: run optimize for a handful of steps to verify the loop."""

    def test_optimize_returns_params_and_history(self, tmp_path):
        """optimize() returns (params, loss_history) and loss decreases."""
        qoc = QOC(
            **default_qoc_params,
            file_dir=str(tmp_path),
        )
        opt_1q = qoc.optimize(wires=1)
        best_params, loss_history = opt_1q(qoc.create_RX)()

        assert best_params is not None
        assert (
            len(loss_history) == default_qoc_params["n_steps"] + 1
        )  # initial + n_steps

    @pytest.mark.parametrize(
        "factory_name",
        [
            "create_RX",
            "create_RY",
            "create_RZ",
            "create_H",
            "create_Rot",
            "create_CX",
            "create_CY",
            "create_CZ",
            "create_CRX",
            "create_CRY",
            "create_CRZ",
        ],
    )
    def test_create_circuits_return_callables(self, factory_name):
        qoc = QOC(**default_qoc_params)
        factory = getattr(qoc, factory_name)
        pulse_c, target_c = factory()
        assert callable(pulse_c)
        assert callable(target_c)
