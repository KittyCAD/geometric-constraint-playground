# This module provides tools to split a geometric constraint system into
# several sequentially soluble blocks, ideally to split a large system
# of equations into smaller, more easily soluble parts.
#
# For example, here we could solve for x1 and y1 first, then use these later - we want
# to identify that.
#
# Variables: x1, y1, x2, y2
# Constraints:
# - C1: x1 = 5               (fixes x1)
# - C2: y1 = 3               (fixes y1)
# - C3: distance(P1,P2) = 10 (relates all four variables)
#
# The current approach is as follows:
#
# 1: From `find_solving_sequence`, it builds a bipartite graph that connects every
# variable to every constraint that references it.
#
# 2: We get the 'maximum matching' on the bipartite graph to try find one constraint for each
# variable; basically trying to find the largest possible set of variable-constraint
# pairs where each variable is paired with exactly one constraint and each constraint
# is paired with at most one variable.
#
# 3: We then build a directed dependency graph from the matching. Each matched
# variable-constraint pair becomes an edge, creating a graph where variables depend
# on the constraints that determine them.
#
# 4: From the directed dependency graph, we look for strongly connected components, where each
# connected component represents a block of variables that must be solved together
# simultaneously. The blocks are ordered by the dependencies between them, and the
# system is solved sequentially.
#
# IT DOES NOT WORK
#
# The fundamental problem (I think) is that when the bipartite graph is not perfectly matchable
# (which happens when the system is over or under-constrained), there are multiple possible
# maximum matchings. NetworkX returns whichever matching it encounters first during its
# traversal, and this depends on Python's hash seed. The selected matching therefore changes
# from run to run, which can reorder the dependency graph and potentially push constraints
# that should be solved first into later blocks. This makes the entire decomposition
# non-deterministic.
#
# Additionally, the maximum matching approach assumes that each variable should be
# determined by exactly one constraint, but geometric constraint systems don't naturally
# have this structure. Variables often participate in multiple constraints, and although
# we still considered all of these during the actual solve, the ordering does not.
#
# Dulmage–Mendelsohn decomposition looks like it might solve these issues, but it looks
# a bit steep for me and it's unclear how big the performance impact would be.
#
# https://www.osti.gov/servlets/purl/1996187
#

from typing import Any, Dict, List, Set

import networkx as nx

from newton.constants import DECOMPOSE_SYSTEM
from newton.constraints import Constraint
from newton.logging_config import logger
from newton.primitives import Point


class StructuralAnalyzer:
    """
    Analyses the structure of a constraint system to find a sequential solving
    order. This is achieved by modeling the system as a bipartite graph of
    constraints and variables, then decomposing it with (I think) Tarjan's algorithm.
    Possibly analogous to block triangularizing the system's Jacobian matrix???
    """

    def __init__(self, constraints: List[Constraint], all_points: List[Point]):
        self.constraints = constraints
        self.all_points = all_points

        # ID or index maps.
        self.point_map = {p.id: p for p in all_points}
        self.constraint_map = {i: c for i, c in enumerate(constraints)}

        # Set of variable names in the system.
        self.variable_names: Set[str] = {
            f"{p.id}_{c}" for p in all_points for c in ("x", "y")
        }

        # Generally our constraints have a scalar residual, but not all.
        self.n_equations = sum(c.get_residual_dim() for c in constraints)

    def find_solving_sequence_full(self) -> List[Dict[str, Any]]:
        # Don't decompose the system into blocks, just return the full system.
        # Used for testing.
        return [
            {
                "points": self.all_points,
                "constraints": list(self.constraints),
            }
        ]

    def find_solving_sequence(self) -> List[Dict[str, Any]]:
        if not DECOMPOSE_SYSTEM:
            return self.find_solving_sequence_full()

        # Decomposes the constraint system into a sequence of soluble blocks.
        if not self.variable_names:
            return []

        if not self.constraints:
            return [{"points": self.all_points, "constraints": []}]

        # Build what we need to get to our dependency graph.
        bipartite_graph = self.build_bipartite_graph()
        variable_to_constraint_map = self.find_matching(bipartite_graph)
        dep_graph = self.build_dependency_graph(variable_to_constraint_map)

        # Then find the strongly connected components (SCCs) in the dependency graph.
        # This should be Tarjan's algorithm; each SCC will be a block of variables
        # that can be solved together.
        scc_list = list(nx.strongly_connected_components(dep_graph))
        ordered_blocks = self.create_ordered_blocks(
            dep_graph, scc_list, variable_to_constraint_map
        )

        logger.info("Found solving sequence using graph decomposition:")
        for i, block in enumerate(ordered_blocks):
            vars_in_block = {f"{p.id}_{c}" for p in block["points"] for c in "xy"}
            num_constraints = sum(c.get_residual_dim() for c in block["constraints"])

            logger.info(
                f" - Block {i + 1}: solves for {sorted(list(vars_in_block))} "
                f"using {num_constraints} constraints"
            )
        return ordered_blocks

    def build_bipartite_graph(self) -> nx.Graph:
        graph = nx.Graph()

        # Add variable nodes.
        graph.add_nodes_from(self.variable_names, bipartite=0)

        # Build and add constraint nodes.
        constraint_nodes = [
            f"C{i}_d{dim}"
            for i, c in self.constraint_map.items()
            for dim in range(c.get_residual_dim())
        ]
        graph.add_nodes_from(constraint_nodes, bipartite=1)

        # Add edges between variables and constraints.
        for i, c in self.constraint_map.items():
            constraint_vars = self.get_vars_for_constraint(c)
            for dim in range(c.get_residual_dim()):
                constraint_node = f"C{i}_d{dim}"
                for var_name in constraint_vars:
                    graph.add_edge(var_name, constraint_node)

        return graph

    def find_matching(self, bipartite_graph: nx.Graph) -> Dict[str, str]:
        # Pairs variables to constraints using 'maximum matching'.
        #
        # Effectively, goes looking for the maximal set of edges where no two edges share a node: "Given a bipartite graph, a matching is a subset of the edges for which every vertex belongs to exactly one of the edges"
        #
        # Ref: https://discrete.openmathbooks.org/dmoi3/sec_matchings.html
        # Ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matching.maximum_matching.html

        # For a system to be structurally 'well-posed', every variable must be uniquely paired with a constraint dimension.

        # A maximum matching finds the largest possible set of edges with no
        # shared nodes. This is the canonical way to test for structural rank.
        matching = nx.bipartite.maximum_matching(
            bipartite_graph, top_nodes=self.variable_names
        )

        # The matching from networkx is bidirectional. We only need one direction.
        variable_to_constraint_map = {
            v: c for v, c in matching.items() if v in self.variable_names
        }

        # For a well-posed system, every single variable must be matched.
        # If even one variable is left out, the system is either over-constrained or
        # under-constrained from a structural standpoint.
        # I don't think we can determine which without Jacobian rank?
        if len(variable_to_constraint_map) < len(self.variable_names):
            message = (
                "System is structurally ill-posed. A perfect matching between "
                "variables and constraints could not be found. This indicates the "
                "problem is either over-constrained or under-constrained."
            )
            logger.warning(message)

        return variable_to_constraint_map

    def build_dependency_graph(
        self, variable_to_constraint_map: Dict[str, str]
    ) -> nx.DiGraph:
        dep_graph = nx.DiGraph()
        dep_graph.add_nodes_from(self.variable_names)

        constraint_to_variable_map = {
            c: v for v, c in variable_to_constraint_map.items()
        }

        for constraint_node, output_variable in constraint_to_variable_map.items():
            # This string splitting is disgusting.
            # TODO: Use a better data structure with access to this.
            constraint_idx = int(constraint_node.split("_")[0][1:])
            constraint = self.constraint_map[constraint_idx]

            all_vars_in_constraint = self.get_vars_for_constraint(constraint)
            input_variables = all_vars_in_constraint - {output_variable}

            for input_var in input_variables:
                dep_graph.add_edge(input_var, output_variable)

        return dep_graph

    def create_ordered_blocks(
        self,
        dep_graph: nx.DiGraph,
        scc_list: List[Set[str]],
        variable_to_constraint_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        # Build up our soluble system (the ordered blocks).

        # "The condensation of G is the graph with each of the strongly connected components contracted into a single node."
        # Ref: https://en.wikipedia.org/wiki/Condensation_(graph_theory)
        # Ref: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.condensation.html
        condensation_graph = nx.condensation(dep_graph, scc=scc_list)
        solve_order_indices = list(nx.topological_sort(condensation_graph))

        ordered_blocks = []
        for node_idx in solve_order_indices:
            variable_block = scc_list[node_idx]

            block_points = self.get_points_from_vars(variable_block)

            # We might not have perfect coverage of constraints in the block.
            constraint_nodes_in_block = {
                variable_to_constraint_map[var]
                for var in variable_block
                if var in variable_to_constraint_map
            }

            constraint_indices = {
                int(c_node.split("_")[0][1:]) for c_node in constraint_nodes_in_block
            }
            block_constraints = [
                self.constraint_map[idx] for idx in sorted(list(constraint_indices))
            ]

            # Feed back all points and constraints in this block.
            ordered_blocks.append(
                {
                    "points": block_points,
                    "constraints": block_constraints,
                }
            )

        return ordered_blocks

    def get_vars_for_constraint(self, constraint: Constraint) -> Set[str]:
        vars_in_constraint = set()
        primitive_ids = constraint.get_involved_primitive_ids()

        for prim_id in primitive_ids:
            if prim_id in self.point_map:
                for coord in ("x", "y"):
                    vars_in_constraint.add(f"{prim_id}_{coord}")

        return vars_in_constraint

    def get_points_from_vars(self, var_set: Set[str]) -> List[Point]:
        point_ids = {var.split("_")[0] for var in var_set}
        points = [self.point_map[pid] for pid in point_ids if pid in self.point_map]
        return points
