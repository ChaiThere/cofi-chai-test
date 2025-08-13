import numpy as np

from . import BaseInferenceTool, error_handler


class MealpySma(BaseInferenceTool):
    r"""Wrapper for Slime Mould Algorithm from mealpy library

    The Slime Mould Algorithm (SMA) is a bio-inspired optimizer that
    mimics the oscillation mode of slime mould in nature. Following CoFI's
    vector abstraction principle, this tool operates purely on flattened
    parameter vectors while supporting spatial regularization at the utility level.

    Based on the paper:
    Li, S., Chen, H., Wang, M., Heidari, A. A., & Mirjalili, S. (2020).
    Slime mould algorithm: A new method for stochastic optimization.
    Future Generation Computer Systems, 111, 300-323.
    """

    documentation_links = [
        "https://mealpy.readthedocs.io/",
        "https://doi.org/10.1016/j.future.2020.03.055",
        "https://github.com/thieu1995/mealpy",
    ]

    short_description = (
        "Slime Mould Algorithm - bio-inspired optimization based on "
        "slime mould foraging behavior (via mealpy)"
    )

    @classmethod
    def required_in_problem(cls) -> set:
        return {"objective", "model_shape"}

    @classmethod
    def optional_in_problem(cls) -> dict:
        return {
            "bounds": None,
            "initial_model": None,
        }

    @classmethod
    def required_in_options(cls) -> set:
        return set()

    @classmethod
    def optional_in_options(cls) -> dict:
        return {
            "algorithm": "OriginalSMA",  # or "DevSMA"
            "epoch": 100,
            "pop_size": 50,
            "pr": 0.03,
            "mode": "single",  # "thread", "process", "swarm"
            "n_workers": None,
            "seed": None,
            "verbose": False,
        }

    @classmethod
    def available_algorithms(cls) -> set:
        return {"OriginalSMA", "DevSMA"}

    def __init__(self, inv_problem, inv_options):
        super().__init__(inv_problem, inv_options)
        self._components_used = list(self.required_in_problem())
        self._setup_problem_dict()
        self._setup_optimizer()

    def _setup_problem_dict(self):
        """Convert CoFI problem to mealpy format

        Following CoFI's vector abstraction principle:
        - model_shape used only for dimensionality (np.prod)
        - Bounds handling supports multiple CoFI formats
        - No spatial awareness at optimizer level
        """
        try:
            from mealpy import FloatVar
        except ImportError:
            raise ImportError(
                "mealpy is required for SMA optimization. "
                "Install it with: pip install mealpy>=3.0.0"
            )

        # Get problem dimensions - CoFI's vector abstraction
        model_shape = self.inv_problem.model_shape
        n_dims = np.prod(model_shape)

        # Setup bounds - handle multiple CoFI formats
        if self.inv_problem.bounds_defined:
            bounds = self.inv_problem.bounds

            # Handle different bound formats from CoFI
            if len(bounds) == 2 and np.isscalar(bounds[0]):
                # Uniform bounds: (-10, 10) for all parameters
                lb = [bounds[0]] * n_dims
                ub = [bounds[1]] * n_dims
            elif len(bounds) == n_dims and all(len(b) == 2 for b in bounds):
                # Per-parameter bounds: [(-10, 10), (-5, 15), ...]
                lb = [b[0] for b in bounds]
                ub = [b[1] for b in bounds]
            else:
                # Legacy format: bounds[0] = lowers, bounds[1] = uppers
                lb = (
                    bounds[0]
                    if isinstance(bounds[0], (list, np.ndarray))
                    else [bounds[0]] * n_dims
                )
                ub = (
                    bounds[1]
                    if isinstance(bounds[1], (list, np.ndarray))
                    else [bounds[1]] * n_dims
                )
        else:
            # Default wide bounds if not specified
            lb = [-100.0] * n_dims
            ub = [100.0] * n_dims

        # Create MEALPY problem dictionary
        self._problem_dict = {
            "obj_func": self._objective_wrapper,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": "min",
        }

    def _objective_wrapper(self, solution):
        """Wrapper to reshape solution for CoFI objective

        Following CoFI's model handling architecture:
        - SMA operates on flattened vectors (MEALPY requirement)
        - CoFI objective functions expect original model_shape
        - This wrapper bridges between the two representations
        - Only extracts scalar objective value (SMA is gradient-free)
        """
        model = solution.reshape(self.inv_problem.model_shape)
        obj_value = self.inv_problem.objective(model)

        # Ensure scalar return even if objective returns (value, gradient) tuple
        # This handles cases where users reuse gradient-based objective functions
        if isinstance(obj_value, (tuple, list)):
            return obj_value[0]
        return obj_value

    def _setup_optimizer(self):
        """Initialize the mealpy optimizer"""
        try:
            from mealpy.bio_based import SMA
        except ImportError:
            raise ImportError(
                "mealpy is required for SMA optimization. "
                "Install it with: pip install mealpy>=3.0.0"
            )

        algorithm = self._params.get("algorithm", "OriginalSMA")

        optimizer_params = {
            "epoch": self._params.get("epoch", 100),
            "pop_size": self._params.get("pop_size", 50),
            "pr": self._params.get("pr", 0.03),
        }

        # Add seed if provided
        if self._params.get("seed") is not None:
            optimizer_params["seed"] = self._params["seed"]

        # Add initial population if initial_model provided
        if self.inv_problem.initial_model_defined:
            initial_flat = self.inv_problem.initial_model.flatten()
            optimizer_params["starting_positions"] = [initial_flat]

        if algorithm == "OriginalSMA":
            self._optimizer = SMA.OriginalSMA(**optimizer_params)
        elif algorithm == "DevSMA":
            self._optimizer = SMA.DevSMA(**optimizer_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def __call__(self) -> dict:
        raw_results = self._call_backend_tool()

        # Format results following CoFI conventions
        result = {
            "success": True,  # SMA doesn't provide explicit success flag
            "model": raw_results.solution.reshape(self.inv_problem.model_shape),
            "objective": raw_results.target.fitness,
            "n_iterations": self._params.get("epoch", 100),
        }

        # Add optimization history if available
        if hasattr(self._optimizer, "history"):
            result["history"] = {
                "objective": self._optimizer.history.list_global_best_fit,
            }

        return result

    @error_handler(
        when="solving optimization problem with Slime Mould Algorithm",
        context="calling mealpy SMA optimizer",
    )
    def _call_backend_tool(self):
        """Run the optimization"""
        mode = self._params.get("mode", "single")
        n_workers = self._params.get("n_workers", None)

        # Run optimization
        if mode == "single":
            g_best = self._optimizer.solve(self._problem_dict)
        else:
            g_best = self._optimizer.solve(
                self._problem_dict, mode=mode, n_workers=n_workers
            )

        return g_best


# CoFI -> Parameter estimation -> Optimization -> Gradient-free optimizers -> mealpy -> SMA
# description: Bio-inspired optimization mimicking slime mould foraging behavior
# documentation: https://mealpy.readthedocs.io/
