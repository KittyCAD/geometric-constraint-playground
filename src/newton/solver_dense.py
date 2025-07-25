from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import newton.backend as nb
from newton.constraints import BaseConstraint
from newton.primitives import Point
from newton.solver_base import DEBUG_LOG, SOLVE_TOLERANCE, Solver2D


class Solver2DDense(Solver2D):
    def solve_subproblem(self, subproblem: Dict[str, Any]):
        # Set the backend to jax for this solve.
        nb.set_backend(nb.Backend.JAX)

        free_points: List[Point] = subproblem["free_points"]
        constraints: List[BaseConstraint] = subproblem["constraints"]

        if DEBUG_LOG:
            print(f"Solving Subproblem: {[p.id for p in free_points]}")

        initial_guess = np.array([[p.x, p.y] for p in free_points]).flatten()

        def get_all_positions_jax(free_vars: jnp.ndarray) -> Dict[str, jnp.ndarray]:
            positions = {p.id: jnp.array([p.x, p.y]) for p in self.points}
            for i, p in enumerate(free_points):
                positions[p.id] = free_vars[i * 2 : i * 2 + 2]
            return positions

        def residuals_vector(free_vars: jnp.ndarray) -> jnp.ndarray:
            positions = get_all_positions_jax(free_vars)
            parts = [c.get_residual(positions) for c in constraints]
            return jnp.concatenate(parts)

        jit_residuals = jax.jit(residuals_vector)
        jit_jacobian = jax.jit(jax.jacfwd(residuals_vector))

        result = least_squares(
            fun=jit_residuals,
            x0=initial_guess,
            jac=jit_jacobian,  # type: ignore
            method="trf",
            xtol=SOLVE_TOLERANCE,
            ftol=SOLVE_TOLERANCE,
            gtol=SOLVE_TOLERANCE,
            verbose=2 if DEBUG_LOG else 0,
        )
        self.update_points_from_result(result, free_points)
