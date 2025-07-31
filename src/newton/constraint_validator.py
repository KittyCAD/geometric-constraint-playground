import itertools
from typing import List, Tuple, Type

import networkx as nx

from newton.constraints import Constraint, LineHorizontal, LineVertical
from newton.exceptions import ConflictError


class ConstraintValidator:
    def run(self, constraints: List[Constraint]) -> List[Constraint]:
        if not constraints:
            return []

        # Build the graph and get the groups of related constraints.
        groups = self.build_constraint_groups(constraints)

        # Validate each group. If any group is invalid, an exception will be raised.
        for group in groups:
            self.process_group(group)

        # If all groups are valid, return the original, unmodified list of constraints.
        return list(constraints)

    def build_constraint_groups(
        self, constraints: List[Constraint]
    ) -> List[List[Constraint]]:
        # Groups constraints by building a graph where nodes are constraints
        # and an edge exists if they share any primitive. The connected components of
        # this graph are the groups.

        graph = nx.Graph()
        # Use constraint indices as nodes for simplicity
        graph.add_nodes_from(range(len(constraints)))

        for i, j in itertools.combinations(range(len(constraints)), 2):
            c1 = constraints[i]
            c2 = constraints[j]

            ids1 = c1.get_involved_primitive_ids()
            ids2 = c2.get_involved_primitive_ids()

            # Add an edge if the sets of primitives overlap.
            if not ids1.isdisjoint(ids2):
                graph.add_edge(i, j)

        # The groups are the connected components of the graph.
        groups = []
        for component in nx.connected_components(graph):
            groups.append([constraints[i] for i in component])

        return groups

    def process_group(self, group: List[Constraint]) -> None:
        # Analyse a group of related constraints for conflicts and redundancies.
        # This method validates the group and raises a ConflictError if any issues are found.

        if len(group) == 1:
            return  # A single constraint is always valid by itself.

        # -----------------------------------------------------------------------------
        # Conflict detection rules.

        # Rule: A line can't be both Horizontal and Vertical.
        # This works because Horizontal(L1) and Vertical(L1) will be in the same group.
        line_ids_with_horizontal = {
            c.line.id for c in group if isinstance(c, LineHorizontal)
        }
        line_ids_with_vertical = {
            c.line.id for c in group if isinstance(c, LineVertical)
        }
        if not line_ids_with_horizontal.isdisjoint(line_ids_with_vertical):
            raise ConflictError(
                "A line cannot be simultaneously Horizontal and Vertical."
            )

        # Rule: No duplicate constraints on the exact same primitives.
        # This treats any direct redundancy as an error.
        seen_instances: set[Tuple[Type[Constraint], frozenset]] = set()
        for c in group:
            instance_key = (type(c), c.get_involved_primitive_ids())
            if instance_key in seen_instances:
                # We found a second constraint of the same type on the same primitives.
                primitive_ids = sorted(list(instance_key[1]))
                raise ConflictError(
                    f"Found duplicate constraint of type '{instance_key[0].__name__}' "
                    f"acting on the same primitives: {primitive_ids}"
                )
            seen_instances.add(instance_key)
        # -----------------------------------------------------------------------------

        return
