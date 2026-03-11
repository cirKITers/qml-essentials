import csv
import os
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from qml_essentials.qoc import (
    Cost,
    CostFnRegistry,
    QOC,
    fidelity_cost_fn,
    pulse_width_cost_fn,
    evolution_time_cost_fn,
)


class TestCost:
    """Unit tests for the Cost wrapper class."""

    def test_scalar_weight(self):
        """Cost with a scalar weight multiplies the result."""
        fn = lambda params: params.sum()
        cost = Cost(cost=fn, weight=2.0)
        result = cost(jnp.array([1.0, 3.0]))
        assert jnp.isclose(result, 8.0)

    def test_tuple_weight(self):
        """Cost with a tuple weight does element-wise weighting and sums."""
        fn = lambda params: (params[0], params[1])
        cost = Cost(cost=fn, weight=(0.5, 0.25))
        result = cost(jnp.array([4.0, 8.0]))
        # 4*0.5 + 8*0.25 = 2 + 2 = 4
        assert jnp.isclose(result, 4.0)

    def test_ckwargs_injection(self):
        """Constant kwargs are injected into the cost function call."""
        fn = lambda params, offset=0.0: params.sum() + offset
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
        fn = lambda params: (params[0],)  # returns 1-tuple
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
        assert meta["default_weight"] == (0.45, 0.45)
        assert "pulse_script" in meta["ckwargs_keys"]

    def test_get_unknown_raises(self):
        """Getting an unregistered name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cost function"):
            CostFnRegistry.get("nonexistent_cost")

    def test_register_and_cleanup(self):
        """Registering a new cost function makes it available."""
        dummy_fn = lambda params: jnp.array(0.0)
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

    def test_parse_cost_arg_with_weight(self):
        """Parsing 'name:w' returns the correct tuple."""
        name, weight = CostFnRegistry.parse_cost_arg("pulse_width:0.3")
        assert name == "pulse_width"
        assert weight == 0.3

    def test_parse_cost_arg_tuple_weight(self):
        """Parsing 'name:w1,w2' returns a tuple weight."""
        name, weight = CostFnRegistry.parse_cost_arg("fidelity:0.6,0.2")
        assert name == "fidelity"
        assert weight == (0.6, 0.2)

    def test_parse_cost_arg_default_weight(self):
        """Omitting the weight part uses the registry default."""
        name, weight = CostFnRegistry.parse_cost_arg("evolution_time")
        assert name == "evolution_time"
        assert weight == 0.075

    def test_parse_cost_arg_unknown_raises(self):
        """Parsing an unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown cost function"):
            CostFnRegistry.parse_cost_arg("bogus:0.5")

    def test_parse_cost_arg_wrong_weight_count_raises(self):
        """Passing wrong number of weights raises ValueError."""
        # fidelity expects 2 weights, giving 1 should fail
        with pytest.raises(ValueError, match="expects 2 weight"):
            CostFnRegistry.parse_cost_arg("fidelity:0.5")

        # pulse_width expects 1 weight, giving 2 should fail
        with pytest.raises(ValueError, match="expects 1 weight"):
            CostFnRegistry.parse_cost_arg("pulse_width:0.3,0.2")


class TestPulseWidthCostFn:
    """Tests for the pulse_width_cost_fn."""

    def test_gaussian_returns_sigma(self):
        """For gaussian envelope the cost equals the sigma parameter."""
        # gaussian: [A, sigma, t] -> n_envelope_params=2, width = p[1]
        params = jnp.array([10.0, 2.5, 0.8])
        cost = pulse_width_cost_fn(params, envelope="gaussian")
        assert jnp.isclose(cost, 2.5)

    def test_general_returns_zero(self):
        """For 'general' envelope (0 envelope params) cost is zero."""
        params = jnp.array([0.5])
        cost = pulse_width_cost_fn(params, envelope="general")
        assert jnp.isclose(cost, 0.0)


class TestEvolutionTimeCostFn:
    """Tests for the evolution_time_cost_fn."""

    def test_at_target_is_zero(self):
        """When t == t_target the cost is exactly zero."""
        params = jnp.array([5.0, 1.0, 1.0])  # t = p[-1] = 1.0
        cost = evolution_time_cost_fn(params, t_target=1.0)
        assert jnp.isclose(cost, 0.0)

    def test_deviation_is_squared_relative(self):
        """The cost equals ((t - t_target) / t_target) ** 2."""
        params = jnp.array([5.0, 1.0, 1.5])  # t = 1.5
        cost = evolution_time_cost_fn(params, t_target=1.0)
        expected = ((1.5 - 1.0) / 1.0) ** 2  # 0.25
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
        """Default cost_fns include fidelity, pulse_width, evolution_time."""
        qoc = QOC()
        names = [name for name, _ in qoc.cost_fns]
        assert "fidelity" in names
        assert "pulse_width" in names
        assert "evolution_time" in names

    def test_custom_cost_fns(self):
        """Custom cost_fns override the defaults."""
        custom = [("fidelity", (0.5, 0.5))]
        qoc = QOC(cost_fns=custom)
        assert qoc.cost_fns == custom

    def test_stores_parameters(self):
        """All __init__ parameters are stored as attributes."""
        qoc = QOC(
            envelope="gaussian",
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
            QOC(cost_fns=[("nonexistent", 0.5)])


class TestSaveResults:
    """Tests for QOC.save_results CSV I/O."""

    def test_creates_new_file(self, tmp_path):
        """save_results creates a new CSV when none exists."""
        qoc = QOC(file_dir=str(tmp_path))
        qoc.save_results("RX", 0.95, [1.0, 2.0, 3.0])

        csv_path = tmp_path / "qoc_results.csv"
        assert csv_path.exists()

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1
        assert rows[0][0] == "RX"
        assert float(rows[0][1]) == pytest.approx(0.95)

    def test_appends_new_gate(self, tmp_path):
        """A second gate is appended as a new row."""
        qoc = QOC(file_dir=str(tmp_path))
        qoc.save_results("RX", 0.9, [1.0, 2.0])
        qoc.save_results("RY", 0.8, [3.0, 4.0])

        with open(tmp_path / "qoc_results.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2
        gate_names = {r[0] for r in rows}
        assert gate_names == {"RX", "RY"}

    def test_overwrites_existing_gate(self, tmp_path):
        """Writing the same gate again overwrites the row."""
        qoc = QOC(file_dir=str(tmp_path))
        qoc.save_results("RX", 0.9, [1.0, 2.0])
        qoc.save_results("RX", 0.95, [5.0, 6.0])

        with open(tmp_path / "qoc_results.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1
        assert rows[0][0] == "RX"
        assert float(rows[0][1]) == pytest.approx(0.95)

    def test_preserves_other_gates_on_overwrite(self, tmp_path):
        """Overwriting one gate does not lose other gates."""
        qoc = QOC(file_dir=str(tmp_path))
        qoc.save_results("RX", 0.9, [1.0])
        qoc.save_results("RY", 0.8, [2.0])
        qoc.save_results("RX", 0.95, [3.0])

        with open(tmp_path / "qoc_results.csv") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2
        gate_map = {r[0]: float(r[1]) for r in rows}
        assert gate_map["RX"] == pytest.approx(0.95)
        assert gate_map["RY"] == pytest.approx(0.8)

    def test_no_file_when_file_dir_is_none(self, tmp_path):
        """When file_dir is explicitly None, nothing is written."""
        qoc = QOC(file_dir=str(tmp_path))
        qoc.file_dir = None
        qoc.save_results("RX", 0.9, [1.0])
        assert not (tmp_path / "qoc_results.csv").exists()


class TestOptimizeSmoke:
    """Smoke test: run optimize for a handful of steps to verify the loop."""

    def test_optimize_returns_params_and_history(self, tmp_path):
        """optimize() returns (params, loss_history) and loss decreases."""
        qoc = QOC(
            envelope="gaussian",
            n_steps=5,
            n_samples=4,
            learning_rate=0.01,
            log_interval=100,
            file_dir=str(tmp_path),
        )
        opt_1q = qoc.optimize(wires=1)
        best_params, loss_history = opt_1q(qoc.create_RX)()

        assert best_params is not None
        assert len(loss_history) == 5 + 1  # initial + n_steps

    def test_create_circuits_return_callables(self):
        """create_RX / create_RY / … return (pulse_circuit, target_circuit)."""
        qoc = QOC()
        for factory in [
            qoc.create_RX,
            qoc.create_RY,
            qoc.create_RZ,
            qoc.create_H,
            qoc.create_Rot,
        ]:
            pulse_c, target_c = factory()
            assert callable(pulse_c)
            assert callable(target_c)

    def test_create_2q_circuits_return_callables(self):
        """Two-qubit create_C* methods return callable pairs."""
        qoc = QOC()
        for factory in [
            qoc.create_CX,
            qoc.create_CY,
            qoc.create_CZ,
            qoc.create_CRX,
            qoc.create_CRY,
            qoc.create_CRZ,
        ]:
            pulse_c, target_c = factory()
            assert callable(pulse_c)
            assert callable(target_c)
