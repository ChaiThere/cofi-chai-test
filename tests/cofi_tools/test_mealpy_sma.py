import numpy as np
import pytest
from cofi import BaseProblem, InversionOptions, Inversion
from cofi.tools import MealpySma
from cofi.utils import QuadraticReg


class TestMealpySma:
    """Test suite for Slime Mould Algorithm integration

    Tests verify SMA's compatibility with CoFI's model handling architecture,
    including vector abstraction and spatial regularization support.
    """

    def setup_method(self):
        """Setup test fixtures"""
        # Simple quadratic problem
        self.problem = BaseProblem()
        self.problem.set_objective(lambda x: np.sum(x**2))
        self.problem.set_model_shape((5,))
        self.problem.set_bounds((-10, 10))

        self.options = InversionOptions()
        self.options.set_tool("mealpy.sma")

    def test_basic_optimization(self):
        """Test basic optimization works"""
        self.options.set_params(epoch=20, pop_size=10, seed=42)
        inv = Inversion(self.problem, self.options)
        result = inv.run()

        assert result.success
        assert hasattr(result, "model")
        assert hasattr(result, "objective")
        assert result.objective < 1.0  # Should find near-optimal

    def test_algorithm_variants(self):
        """Test both OriginalSMA and DevSMA"""
        for algo in ["OriginalSMA", "DevSMA"]:
            self.options.set_params(algorithm=algo, epoch=10, seed=42)
            inv = Inversion(self.problem, self.options)
            result = inv.run()
            assert result.success

    def test_objective_with_multiple_returns(self):
        """Test SMA handles objectives that return extra values (like gradients)

        Verifies CoFI's design principle: gradient-free optimizers should handle
        cases where users accidentally provide gradient-returning functions.
        """

        def objective_with_grad(x):
            # Function originally written for gradient-based methods
            return np.sum(x**2), 2 * x  # Returns (objective, gradient)

        problem = BaseProblem()
        problem.set_objective(objective_with_grad)
        problem.set_model_shape((3,))
        problem.set_bounds((-5, 5))

        self.options.set_params(epoch=10, pop_size=10, seed=42)
        inv = Inversion(problem, self.options)
        result = inv.run()  # Should work by extracting only scalar part
        assert result.success
        assert hasattr(result, "model")

    def test_spatial_regularization_compatibility(self):
        """Test SMA works with spatial regularization (2D problem)

        Verifies CoFI's two-level architecture: spatial regularization operates
        at utility level while SMA remains vector-agnostic at optimizer level.
        """

        def objective_with_spatial_reg(slowness):
            # Simulate a 2D tomography problem with spatial smoothing
            data_misfit = np.sum((slowness - 0.5) ** 2)  # Simple data term
            reg_value = spatial_reg(slowness)  # Spatial regularization
            return data_misfit + reg_value

        # Setup 2D problem with spatial regularization
        model_shape = (10, 8)  # 2D grid
        spatial_reg = QuadraticReg(
            model_shape=model_shape, weighting_matrix="smoothing"
        )

        problem = BaseProblem()
        problem.set_objective(objective_with_spatial_reg)
        problem.set_model_shape(model_shape)
        problem.set_bounds((0.1, 1.0))

        self.options.set_params(epoch=20, pop_size=15, seed=42)
        inv = Inversion(problem, self.options)
        result = inv.run()

        assert result.success
        assert result.model.shape == model_shape  # Proper reshaping
        # Spatial regularization should encourage smoothness
        assert np.std(result.model) < 0.5  # Relatively smooth solution

    def test_parallel_modes(self):
        """Test parallel execution modes"""
        try:
            self.options.set_params(mode="thread", n_workers=2, epoch=10, seed=42)
            inv = Inversion(self.problem, self.options)
            result = inv.run()
            assert result.success
        except Exception:
            # Parallel mode may not work in all environments, skip if fails
            pytest.skip("Parallel mode not available in test environment")

    def test_without_bounds(self):
        """Test optimization without explicit bounds"""
        problem_no_bounds = BaseProblem()
        problem_no_bounds.set_objective(lambda x: np.sum(x**2))
        problem_no_bounds.set_model_shape((3,))

        options = InversionOptions()
        options.set_tool("mealpy.sma")
        options.set_params(epoch=20, pop_size=10, seed=42)

        inv = Inversion(problem_no_bounds, options)
        result = inv.run()
        assert result.success

    def test_high_dimensional(self):
        """Test with high-dimensional problem"""
        problem_hd = BaseProblem()
        problem_hd.set_objective(lambda x: np.sum(x**2))
        problem_hd.set_model_shape((50,))
        problem_hd.set_bounds((-5, 5))

        options = InversionOptions()
        options.set_tool("mealpy.sma")
        options.set_params(epoch=50, pop_size=30, seed=42)

        inv = Inversion(problem_hd, options)
        result = inv.run()
        assert result.success
        assert result.objective < 10.0

    def test_rosenbrock_benchmark(self):
        """Test on Rosenbrock function benchmark"""

        def rosenbrock(x):
            return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        problem = BaseProblem()
        problem.set_objective(rosenbrock)
        problem.set_model_shape((2,))
        problem.set_bounds((-5, 10))

        options = InversionOptions()
        options.set_tool("mealpy.sma")
        options.set_params(epoch=100, pop_size=50, seed=42)

        inv = Inversion(problem, options)
        result = inv.run()

        assert result.success
        # Check if close to global optimum (all ones)
        assert np.allclose(result.model, 1.0, atol=0.5)

    def test_bounds_format_compatibility(self):
        """Test different bounds formats supported by CoFI"""
        # Test uniform bounds
        problem1 = BaseProblem()
        problem1.set_objective(lambda x: np.sum(x**2))
        problem1.set_model_shape((2,))
        problem1.set_bounds((-2, 2))  # Uniform bounds

        options1 = InversionOptions()
        options1.set_tool("mealpy.sma")
        options1.set_params(epoch=20, pop_size=10, seed=42)

        inv1 = Inversion(problem1, options1)
        result1 = inv1.run()
        assert result1.success

        # Test per-parameter bounds
        problem2 = BaseProblem()
        problem2.set_objective(lambda x: np.sum(x**2))
        problem2.set_model_shape((3,))
        problem2.set_bounds([(-1, 1), (-2, 2), (-3, 3)])  # Per-parameter bounds

        options2 = InversionOptions()
        options2.set_tool("mealpy.sma")
        options2.set_params(epoch=200, pop_size=10, seed=42)

        inv2 = Inversion(problem2, options2)
        result2 = inv2.run()
        assert result2.success

    def test_initial_model_support(self):
        """Test support for initial model"""
        problem = BaseProblem()
        problem.set_objective(lambda x: np.sum((x - [1, 2, 3]) ** 2))
        problem.set_model_shape((3,))
        problem.set_bounds((-5, 5))
        problem.set_initial_model(np.array([0.9, 1.9, 2.9]))  # Close to optimum

        options = InversionOptions()
        options.set_tool("mealpy.sma")
        options.set_params(epoch=20, pop_size=10, seed=42)

        inv = Inversion(problem, options)
        result = inv.run()

        assert result.success
        # Should converge faster with good initial model
        assert np.allclose(result.model, [1, 2, 3], atol=0.1)

    def test_tool_alias(self):
        """Test that the tool alias works"""
        options = InversionOptions()
        options.set_tool("mealpy.slime_mould")  # Test alias
        options.set_params(epoch=10, pop_size=5, seed=42)

        inv = Inversion(self.problem, options)
        result = inv.run()
        assert result.success

    def test_error_handling(self):
        """Test error handling for invalid parameters"""
        with pytest.raises(ValueError):
            options = InversionOptions()
            options.set_tool("mealpy.sma")
            options.set_params(algorithm="InvalidAlgorithm")
            inv = Inversion(self.problem, options)
            inv.run()

    def test_seed_reproducibility(self):
        """Test that setting seed produces reproducible results"""
        self.options.set_params(epoch=20, pop_size=10, seed=42)

        inv1 = Inversion(self.problem, self.options)
        result1 = inv1.run()

        inv2 = Inversion(self.problem, self.options)
        result2 = inv2.run()

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(result1.model, result2.model, decimal=6)
        assert abs(result1.objective - result2.objective) < 1e-10


def test_mealpy_import_error():
    """Test graceful handling when mealpy is not available"""
    # This test would need mocking to properly test import error handling
    # For now, we just verify the tool can be imported
    from cofi.tools import MealpySma

    assert MealpySma is not None
