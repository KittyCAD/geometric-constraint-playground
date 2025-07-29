from typing import Any, Dict, List, Sequence, Set

import networkx as nx

from newton.constraints import BaseConstraint
from newton.exceptions import ConflictError
from newton.primitives import Point


class StructuralAnalyzer:
    """
    Analyses the structure of a constraint system to find a sequential solving
    order. This is achieved by modeling the system as a bipartite graph of
    constraints and variables, then decomposing it with (I think) Tarjan's algorithm.
    Possibly analogous to block triangularizing the system's Jacobian matrix???
    """

    def __init__(self, constraints: Sequence[BaseConstraint], all_points: List[Point]):
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

    def find_solving_sequence(self) -> List[Dict[str, Any]]:
        # Decomposes the constraint system into a sequence of soluble blocks.
        if not self.variable_names:
            return []

        if not self.constraints:
            return [{"points": self.all_points, "constraints": []}]

        if len(self.variable_names) != self.n_equations:
            raise ConflictError(
                f"The system is structurally unsound. There are {len(self.variable_names)} unknowns "
                f"but {self.n_equations} equations (degrees of freedom required by constraints)."
            )

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

        print("Found solving sequence using graph decomposition:")
        for i, block in enumerate(ordered_blocks):
            vars_in_block = {f"{p.id}_{c}" for p in block["points"] for c in "xy"}
            num_constraints = sum(c.get_residual_dim() for c in block["constraints"])

            print(
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
            raise ConflictError(
                "System is structurally ill-posed. A perfect matching between "
                "variables and constraints could not be found. This indicates the "
                "problem is either over-constrained or under-constrained."
            )

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
            constraint_nodes_in_block = {
                variable_to_constraint_map[var] for var in variable_block
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
        points = [self.point_map[pid] for pid in point_ids if pid in self.point_map]
        return points
