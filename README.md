# geometric-constraint-playground

This repo contains experimental code for exploring geometric constraint solving problems. It is not
intended for production use; it is intended to be used to help establish which approaches are most
effective for solving the sort of problems we expect to encounter within the sketcher.

## Problem

The problem is to solve a set of geometric constraints, e.g.,

- 'Point A and Point B are coincident.'
- 'Line L1 is parallel to Line L2.'
- 'Circle C1 is tangent to Circle C2.'

Practically, this problem is solved via numerical methods. For each constraint, we derive a residual
function that returns how distant from being satisfied the constraint is. For example, for a
distance constraint, the residual would be `current_distance - target_distance`. When all
constraints are satisfied, all residuals will be zero (within some tolerance).

This approach lets us frame the problem as a multivariable root-finding problem, which we solve
using a non-linear least-squares algorithm.

## Structural optimisation

The solve can be 'structurally' optimised in two ways that we care about:

- **Separating decoupled systems:** In graph terms, this is finding the connected components of the
  constraint graph. If the graph separates into multiple components, we can solve each smaller
  system independently, which is more efficient.
- **Symbolic variable substitution:** For equality constraints like 'Point A is coincident with
  Point B', we can eliminate one of the variables from the problem entirely. All references to Point
  B's coordinates can be replaced with references to Point A's coordinates, reducing the total
  number of variables the solver must handle.

### Quasi-independent subsystems

I spent a lot of cycles trying to identify ways to partition quasi-independent subsystems (i.e.,
block triangular decomposition of the Jacobian). For example, with this sort of system:

```
3x + y = 11
2x + y = 8
4x + 2y - z = 11


Which in matrix form would be:


[3 1 0]  [x]   [11]
[2 1 0]  [y] = [8]
[4 2 -1] [z]   [11]
```

It is possible to partition the system into two quasi-independent subsystems. First you solve for x
and y, then you substitute those values into the third equation to solve for z.

```
[3 1 0]  [x]   [11]
[2 1 0]  [y] = [8]
```

Then:

```
4x + 2y - z = 11
12 + 4 - z = 11
-z = -5

Which in matrix form would be:

[-1] [z] = [-5]
```

(See: `scratch/system_partitioning.py` for this particular example.)

However, it appears that finding a general algorithm for this, like the
[Dulmage-Mendelsohn decomposition](https://en.wikipedia.org/wiki/Dulmage%E2%80%93Mendelsohn_decomposition),
is complex and offers little benefit over a modern sparse solver.

It also seems that sparse solvers are well-optimised for dealing with the partially decoupled
systems that exist in this case; they exploit the zeros in the matrix to avoid unnecessary
calculations.

This does, however, highlight the likely importance of using a suitable sparse solver. It seems that
a good sparse solver will make significant use of this sort of structural aspect of the problem.

## Solve method

The core of the solver currently uses `scipy.optimize.least_squares` with the Trust Region
Reflective (`trf`) algorithm because it plays nicely with a sparse Jacobian. However, we should be
able to use a quasi-Newton method so long as it has good sparse support.

The `trf` method is an iterative method supposedly well-suited for non-linear least-squares
problems. At each iteration, the solver computes the Jacobian matrix (the matrix of all partial
derivatives of residuals with respect to all variables) and solves a linear subproblem to find the
next 'guess' for the variable values that minimizes the residuals.

The underlying linear solve at each step is analogous to solving the normal equation
$(J^T J)\delta x = -J^T r$. To handle underdetermined systems, we use Tikhonov regularisation, which
modifies the equation to $(J^T J + \lambda^2 I)\delta x = -J^T r$.

The $\lambda^2 I$ term ensures the matrix is invertible and stabilises the solution. This pushes the
solver toward a solution that has a minimal deviation from its initial state, effectively finding
the 'closest' (minimum-norm) valid solution (wrt. initial positions) when degrees of freedom exist.

The system's state (underdetermined, overdetermined, or fully determined) is assessed via the rank
of the Jacobian matrix before the solve begins, computed via either singular value decomposition
(SVD) or QR decomposition.

We do not currently compute rank dynamically during the solve; I don't have a good feel for how
expensive this would be.

## Todos and caveats for the reader

- **Primitive support:** The solver currently only reasons about free `Point` primitives, even when
  they are part of a `Line`. Extending the framework to other geometric primitives like arcs or
  circles would require defining their variables (e.g., center, radius) and implementing the
  corresponding residual and Jacobian functions. Several `UnsupportedPrimitiveError` exceptions in
  the code highlight this shortcoming.
- **Symbolic substitution:** The substitution logic is currently limited to a few point-based
  equality constraints. To be fully effective, this should be expanded to handle more scenarios
  where variables can be eliminated.
- **Solver choice:** The codebase includes both a sparse solver (manual Jacobians) and a dense
  solver (using JAX for auto-differentiation). A sparse solver approach seems like it should be
  significantly more performant for this problem domain and we should index hard on that.
  - Sympy was used extensively to arrive at our manual derivatives; see `derivative_deriver.py`.

## Usage

To get set up with the project, you need [uv](https://docs.astral.sh/uv/).

This is wholly untested but I think the setup should be something like:

```bash
cd geometric-constraint-playground
uv sync
```

Then, to run the example in the virtual environment:

```bash
source .venv/bin/activate
cd src
python -m example
```

There are currently three example functions in `example.py`:

- `constrain_rectangles()`: Constrains two rectangles; these are wholly decoupled systems and the
  solver should identify two independent subsystems.
- `constrain_parallel_offset()`: Constrains two lines to be parallel and offset by a fixed distance.
- `constrain_underdetermined()`: Constrains a sort of arm linkage, with its final point being
  underdetermined. The solver should arrive at a minimum-norm solution that is closest to the
  initial guess.
  - Note that this system also allows for symbolic substitution of the final point, which should be
    reported if a suitably noisy log level is set.

Unfortunately, for the time being, you have to modify the `example.py` file to alter which functions
are called and choose whether to plot the results.

Additionally, `constants.py` contains some configuration options that you can modify to change the
solver's behaviour, such as whether to use a sparse or dense solver, whether to use symbolic
substitution, and the convergence tolerance for the solver.

![Example Output](images/example-01.png)

![Example Output](images/example-02.png)

![Example Output](images/example-03.png)

Note the mention of symbolic substitution in the log output of the `constrain_underdetermined()`
example:

![Example Output](images/example-04.png)
