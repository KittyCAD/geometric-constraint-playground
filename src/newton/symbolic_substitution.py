from typing import Dict, List, Set

from newton.constants import EPS
from newton.constraints import (
    Constraint,
    LineHorizontal,
    LinesParallel,
    LineVertical,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import logger
from newton.primitives import Line, Point


def find(var_id: str, parent_map: Dict[str, str]) -> str:
    # Find the root of the set.
    root = var_id
    while root in parent_map:
        root = parent_map[root]

    # Make every node in the path point directly to the root.
    current = var_id
    while current in parent_map:
        parent = parent_map[current]
        parent_map[current] = root
        current = parent

    return root


def union(var1_id: str, var2_id: str, parent_map: Dict[str, str]):
    root1 = find(var1_id, parent_map)
    root2 = find(var2_id, parent_map)

    if root1 != root2:
        # Simple union: make one root the parent of the other.
        parent_map[root1] = root2


def get_substituted_point(
    p: Point, parent_map: Dict[str, str], point_map: Dict[str, Point]
) -> Point:
    # Returns a new Point object pointing to the surrogate IDs.
    x_var_id = f"{p.id}_x"
    y_var_id = f"{p.id}_y"

    # Find the surrogate ID for the x and y coordinates
    x_repr_id = find(x_var_id, parent_map).split("_")[0]
    y_repr_id = find(y_var_id, parent_map).split("_")[0]

    # If the surrogate is the same as the original, return the original point.
    if x_repr_id == p.id and y_repr_id == p.id:
        return p

    # If they've been substituted, return the surrogate Point objects.
    # Note: This assumes we can substitute x and y to different points,
    # which is possible with X/Y distance constraints.
    repr_point_id = find(x_var_id, parent_map).split("_")[0]
    return point_map[repr_point_id]


def rewrite_constraint(
    c: Constraint, parent_map: Dict[str, str], point_map: Dict[str, Point]
) -> Constraint:
    # Creates a new constraint object with substituted primitives.
    match c:
        case (
            PointPointEuclideanDistance()
            | PointPointXDistance()
            | PointPointYDistance()
        ):
            p1_sub = get_substituted_point(c.p1, parent_map, point_map)
            p2_sub = get_substituted_point(c.p2, parent_map, point_map)
            return type(c)(p1=p1_sub, p2=p2_sub, distance=c.distance)

        case LineHorizontal() | LineVertical():
            p1_sub = get_substituted_point(c.line.p1, parent_map, point_map)
            p2_sub = get_substituted_point(c.line.p2, parent_map, point_map)
            new_line = Line(p1=p1_sub, p2=p2_sub, id=c.line.id)
            return type(c)(line=new_line)

        case LinesParallel():
            l1_p1_sub = get_substituted_point(c.line1.p1, parent_map, point_map)
            l1_p2_sub = get_substituted_point(c.line1.p2, parent_map, point_map)
            l2_p1_sub = get_substituted_point(c.line2.p1, parent_map, point_map)
            l2_p2_sub = get_substituted_point(c.line2.p2, parent_map, point_map)
            new_line1 = Line(p1=l1_p1_sub, p2=l1_p2_sub, id=c.line1.id)
            new_line2 = Line(p1=l2_p1_sub, p2=l2_p2_sub, id=c.line2.id)
            return LinesParallel(line1=new_line1, line2=new_line2)

        case _:
            # Default case - return the original constraint
            return c

    # TODO: Add other constraint types.

    return c


def perform_symbolic_substitution(
    constraints: List[Constraint], points: List[Point]
) -> tuple[List[Constraint], Dict[str, str]]:
    """
    Performs a symbolic substitution pass on a constraint system.

    This function finds simple equality constraints (e.g., two points being coincident)
    and uses them to eliminate variables from the system. It does this by
    rewriting the remaining constraints to use a single surrogate variable
    for each set of equivalent variables.

    Returns: (simplified_constraints, point_id_mapping)
    where point_id_mapping maps original_point_id -> simplified_point_id
    """
    point_map = {p.id: p for p in points}

    # https://www.geeksforgeeks.org/dsa/introduction-to-disjoint-set-data-structure-or-union-find-algorithm/
    parent_map: Dict[str, str] = {}
    substitution_map: Dict[str, str] = {}

    constraints_to_skip: Set[int] = set()

    # Build the equivalence sets using 'Union-Find'.
    # Build the equivalence sets using 'Union-Find'.
    for i, c in enumerate(constraints):
        match c:
            case PointPointEuclideanDistance() if c.distance < EPS:
                p1_id, p2_id = c.p1.id, c.p2.id
                union(f"{p1_id}_x", f"{p2_id}_x", parent_map)
                union(f"{p1_id}_y", f"{p2_id}_y", parent_map)
                constraints_to_skip.add(i)

            case PointPointXDistance() if c.distance < EPS:
                p1_id, p2_id = c.p1.id, c.p2.id
                union(f"{p1_id}_x", f"{p2_id}_x", parent_map)
                constraints_to_skip.add(i)

            case PointPointYDistance() if c.distance < EPS:
                p1_id, p2_id = c.p1.id, c.p2.id
                union(f"{p1_id}_y", f"{p2_id}_y", parent_map)
                constraints_to_skip.add(i)

    # Build the final substitution map from variable to surrogate.
    all_var_ids = {f"{p.id}_{coord}" for p in point_map.values() for coord in "xy"}
    for var_id in all_var_ids:
        root = find(var_id, parent_map)
        if root != var_id:
            substitution_map[var_id] = root

    if not substitution_map:
        logger.info("No substitutions found. Returning original constraints.")
        return constraints, {}

    logger.info(f"Found {len(substitution_map)} substitutions to perform.")

    # Rewrite the constraint list
    simplified_constraints: List[Constraint] = []
    for i, c in enumerate(constraints):
        if i in constraints_to_skip:
            continue

        new_c = rewrite_constraint(c, parent_map, point_map)
        simplified_constraints.append(new_c)

    # Build the final point mapping
    point_id_mapping = {}
    for point in points:
        x_root = find(f"{point.id}_x", parent_map).split("_")[0]
        y_root = find(f"{point.id}_y", parent_map).split("_")[0]

        # For now, assume x and y map to the same simplified point
        # (your substitution logic should ensure this for valid geometric constraints)
        assert x_root == y_root, (
            f"Point {point.id} has split coordinates: x→{x_root}, y→{y_root}"
        )

        point_id_mapping[point.id] = x_root

    return simplified_constraints, point_id_mapping
