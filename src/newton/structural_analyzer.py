import itertools
from typing import Any, Dict, List, Sequence, Set

from newton.constraints import BaseConstraint
from newton.primitives import Point


class StructuralAnalyzer:
    """
    Analyses the structure of a constraint system to find soluble sublocks by iteratively
    identifying independently soluble subsystems.
    """

    def __init__(self, constraints: Sequence[BaseConstraint], free_points: List[Point]):
        self.constraints = constraints
        self.free_points = free_points
        self.point_map = {p.id: p for p in free_points}
        self.variable_names = {f"{p.id}_{c}" for p in free_points for c in ("x", "y")}

        # Create a map from the index to the constraint object.
        self.constraint_map = {i: c for i, c in enumerate(constraints)}

    def find_solving_sequence(self) -> List[Dict[str, Any]]:
        """
        Decomposes the constraint system into a sequence of soluble blocks.
        """
        remaining_constraint_indices = list(self.constraint_map.keys())
        known_vars: Set[str] = set()
        ordered_blocks = []

        # Iterate over the constraints build some soluble blocks.
        while True:
            made_progress = False

            constraint_idx_to_unknowns: Dict[int, Set[str]] = {
                i: self.get_vars_for_constraint(self.constraint_map[i]) - known_vars
                for i in remaining_constraint_indices
            }

            for size in range(1, len(remaining_constraint_indices) + 1):
                # Combine indices.
                for index_subset in itertools.combinations(
                    remaining_constraint_indices, size
                ):
                    vars_in_subset = set().union(
                        *(constraint_idx_to_unknowns[i] for i in index_subset)
                    )

                    if len(index_subset) == len(vars_in_subset):
                        # Block found!
                        block_points = self.get_points_from_vars(vars_in_subset)

                        # Look up the constraint objects using their indices.
                        block_constraints = [
                            self.constraint_map[i] for i in index_subset
                        ]

                        ordered_blocks.append(
                            {
                                "free_points": block_points,
                                "constraints": block_constraints,
                            }
                        )

                        known_vars.update(vars_in_subset)

                        # Update the list of remaining indices.
                        remaining_constraint_indices = [
                            i
                            for i in remaining_constraint_indices
                            if i not in index_subset
                        ]

                        made_progress = True
                        break

                if made_progress:
                    break

            if not made_progress:
                break

        if remaining_constraint_indices:
            remaining_vars = self.variable_names - known_vars
            remaining_points = self.get_points_from_vars(remaining_vars)

            remaining_constraints = [
                self.constraint_map[i] for i in remaining_constraint_indices
            ]
            ordered_blocks.append(
                {
                    "free_points": remaining_points,
                    "constraints": remaining_constraints,
                }
            )

        print("Found solving sequence:")
        for i, block in enumerate(ordered_blocks):
            vars_in_block = {f"{p.id}_{c}" for p in block["free_points"] for c in "xy"}
            print(
                f" - Block {i + 1}: solves for {sorted(list(vars_in_block))} using {len(block['constraints'])} constraints"
            )

        return ordered_blocks

    def get_vars_for_constraint(self, constraint: BaseConstraint) -> Set[str]:
        vars_in_constraint = set()
        primitive_ids = constraint.get_involved_primitive_ids()
        for prim_id in primitive_ids:
            if prim_id in self.point_map:
                for coord in ("x", "y"):
                    vars_in_constraint.add(f"{prim_id}_{coord}")
        return vars_in_constraint

    def get_points_from_vars(self, var_set: Set[str]) -> List[Point]:
        point_ids = {var.split("_")[0] for var in var_set}
        return [self.point_map[pid] for pid in point_ids if pid in self.point_map]
