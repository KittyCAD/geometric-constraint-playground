# geometric-constraint-playground

This repo contains experimental code for exploring geometric constraint solving problems. It is not
intended for production use; it is intended to be used to help establish which approaches are most
effective for solving the sort of problems we expect to encounter within the sketcher.

## Problem

The problem is to solve a set of geometric constraints, e.g.,

- "Point A and Point B are coincident."
- "Line L1 is parallel to Line L2."
- "Circle C1 is tangent to Circle C2."

Practically, this problem will be solved via numerical methods. For each constraint, we can derive a
function that describes the constraint in terms of the variables involved. For example, the
constraint "Point A and Point B are coincident" can be expressed as a function that returns the
distance between the two points, which should be (within some reasonable tolerance) zero when the
constraint is satisfied.

This approach of building a set of residual functions let's us then adopt a multivariable
root-finding approach to solve the problem. Because of the requirement for geometric constraints
that involve trigonometric functions and other non-linear functions, this pushes us down the path of
quasi-Newton methods.

## Structural optimisation

The solve can be 'structurally' optimised in two ways that we care about:

- Separating the system of constraints into independent/decoupled blocks.
  - In graph terms, this is separating the connected components of the constraint graph.
    Practically, this allows for the identification of block diagonal components (in this case in
    the Jacobian).
  - If we have more than one connected component, we can solve each component independently.
- Performing substitution/surrogate variable insertion.
  - For example, where we have a constraint like "Point A is coincident with Point B", we can
    replace all references to either point with the other, or with some surrogate which represents
    them both.

### Quasi-independent subsystems

I spent a lot of cycles trying to identify possible ways to partition the independent subsystems to
help solve. For example, with this sort of system:

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

However, it appears that there is not a general way to do this. There exist approaches like
[Dulmage-Mendelsohn decomposition](https://en.wikipedia.org/wiki/Dulmage%E2%80%93Mendelsohn_decomposition)
([see also](https://www.osti.gov/servlets/purl/1996187)) that sound relevant, but all of my
experimentation suggested this is not an avenue worth sinking more time into. Ultimately, the system
forms a single connected component, and thus the variables are inherently coupled.

It also seems that sparse solvers are well-optimised for dealing with the partially decoupled
systems that exist in this case; they exploit the zeros in the matrix to avoid unnecessary
calculations.

This does, however, highlight the likely importance of using a suitable sparse solver. It seems that
a good sparse solver will make significant use of this sort of structural aspect of the problem.

# Todos and caveats for the reader

- We currently only deal with free points, even if these are defined as part of a line primitive.
  - We do not deal with, e.g., arcs where we might need a more complex representation of the
    primitive.
- Although we reference free 'primitives', there are several `UnsupportedPrimitiveError` exceptions
  in the code that indicate that we are not yet handling all primitive types. The code is currently
  only set up to handle `Point` primitives.
- The symbolic substitution only deals with a small set of the scenarios where this could be
  applied. We'll need to add more if this sort of approach is to be useful in production.
