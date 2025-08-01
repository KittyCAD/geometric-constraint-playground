import networkx as nx
import numpy as np
from scipy.optimize import least_squares


def analyze_connectivity(A):
    # Create a bipartite graph: equations (rows) and variables (columns).
    graph = nx.Graph()

    n_equations, n_variables = A.shape

    # Add nodes for equations.
    equation_nodes = [f"e{i}" for i in range(n_equations)]
    variable_nodes = [f"v{j}" for j in range(n_variables)]

    graph.add_nodes_from(equation_nodes, bipartite=0)
    graph.add_nodes_from(variable_nodes, bipartite=1)

    # Add edges where A[i,j] != 0, i.e. equation i involves variable j.
    for i in range(n_equations):
        for j in range(n_variables):
            if A[i, j] != 0:
                graph.add_edge(f"e{i}", f"v{j}")

    # Find connected components.
    components = list(nx.connected_components(graph))
    n_components = len(components)

    print(f"Number of connected components: {n_components}")

    # Do adjacency matrix.
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    print("Adjacency matrix of the bipartite graph:")
    print(adjacency_matrix)

    return


def solve(A, b):
    # You've just gotta s̶e̶n̶d̶  solve it.
    def residuals(x):
        return A @ x - b

    x0 = np.ones(A.shape[1])
    result = least_squares(residuals, x0)
    return result.x


def solve_partitioned(A, b):
    # This is how I would do this by hand basically.

    # Whole system solution: [x, y, z].
    soln_global = np.zeros(3)

    # Carve out the first two equations and unknowns.
    A_part_1 = A[:2, :2]
    b_part_1 = b[:2]

    def residuals_partition(xy_values):
        return A_part_1 @ xy_values - b_part_1

    x0_part1 = np.ones(A_part_1.shape[1])

    # Solve for x and y.
    result_part_1 = least_squares(residuals_partition, x0_part1)

    # Fill in the x and y values in our solution vector.
    soln_global[0] = result_part_1.x[0]  # x
    soln_global[1] = result_part_1.x[1]  # y

    # Now forward substitute to find z using.
    # Set up the second part as a 1x1 matrix problem.
    known_part = A[2, :2] @ soln_global[:2]
    A_part_2 = A[2, 2]
    b_part_2 = b[2] - known_part

    # These need to be array to get matrix math out of them.
    A_part_2 = np.array([[A_part_2]])
    b_part_2 = np.array([b_part_2])

    def residuals_partition_2(z_values):
        return A_part_2 @ z_values - b_part_2

    x0_part_2 = np.ones(A_part_2.shape[1])

    # Solve for z using least squares.
    result_part_2 = least_squares(residuals_partition_2, x0_part_2)

    # Fill in the z value.
    soln_global[2] = result_part_2.x[0]

    return soln_global


if __name__ == "__main__":
    # Example system of equations:
    # 3x + y = 11
    # 2x + y = 8
    # 4x + 2y - z = 11

    A = np.array([[3, 1, 0], [2, 1, 0], [4, 2, -1]])
    b = np.array([11, 8, 11])

    analyze_connectivity(A)

    print("Full least squares solution ['disconnected system']:")
    solution1 = solve(A, b)
    print(solution1)

    print("Partitioned solution ['independently soluble subsystem']:")
    solution2 = solve_partitioned(A, b)
    print(f"Final result: {solution2}")
