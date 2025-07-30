import sympy as sp

# Use pretty printing for clear console output.
sp.init_printing(use_unicode=True)


def compute_point_fixed_derivatives():
    px, py = sp.symbols("px py")
    fx, fy = sp.symbols("fx fy")

    # Two residuals: R1 = px - fx, R2 = py - fy
    residual_x = px - fx
    residual_y = py - fy

    print("Residuals: R1 = px - fx, R2 = py - fy")
    print(f"∂R1/∂px = {sp.diff(residual_x, px)}")
    print(f"∂R1/∂py = {sp.diff(residual_x, py)}")
    print(f"∂R2/∂px = {sp.diff(residual_y, px)}")
    print(f"∂R2/∂py = {sp.diff(residual_y, py)}")


def compute_point_point_euclidean_distance_derivatives():
    x1, y1, x2, y2, d = sp.symbols("x1 y1 x2 y2 d")
    residual = sp.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) - d

    variables = [x1, y1, x2, y2]

    print("Residual: R = sqrt((x1-x2)² + (y1-y2)²) - d")

    for var in variables:
        deriv = sp.diff(residual, var)
        simplified_deriv = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified_deriv}")


def compute_point_point_x_distance_derivatives():
    x1, x2, d = sp.symbols("x1 x2 d")

    dist_raw = x1 - x2

    # Define the residual using a piecewise function for clearer derivatives
    residual = sp.Piecewise(
        (dist_raw - d, dist_raw >= 0), (-dist_raw - d, dist_raw < 0)
    )

    print("Residual: R = |x1 - x2| - d")
    print("When (x1 - x2) >= 0:")
    print("  ∂R/∂x1 = 1")
    print("  ∂R/∂x2 = -1")
    print("When (x1 - x2) < 0:")
    print("  ∂R/∂x1 = -1")
    print("  ∂R/∂x2 = 1")

    # Also show the symbolic derivatives for completeness.
    print("\nSymbolic derivatives:")
    print(f"∂R/∂x1 = {sp.diff(residual, x1)}")
    print(f"∂R/∂x2 = {sp.diff(residual, x2)}")


def compute_point_point_y_distance_derivatives():
    y1, y2, d = sp.symbols("y1 y2 d")

    dist_raw = y1 - y2

    # Define the residual using a piecewise function for clearer derivatives
    residual = sp.Piecewise(
        (dist_raw - d, dist_raw >= 0), (-dist_raw - d, dist_raw < 0)
    )

    print("Residual: R = |y1 - y2| - d")
    print("When (y1 - y2) >= 0:")
    print("  ∂R/∂y1 = 1")
    print("  ∂R/∂y2 = -1")
    print("When (y1 - y2) < 0:")
    print("  ∂R/∂y1 = -1")
    print("  ∂R/∂y2 = 1")

    # Also show the symbolic derivatives for completeness.
    print("\nSymbolic derivatives:")
    print(f"∂R/∂y1 = {sp.diff(residual, y1)}")
    print(f"∂R/∂y2 = {sp.diff(residual, y2)}")


def compute_line_horizontal_derivatives():
    y1, y2 = sp.symbols("y1 y2")
    residual = y1 - y2

    print("Residual: R = y1 - y2")
    print(f"∂R/∂y1 = {sp.diff(residual, y1)}")
    print(f"∂R/∂y2 = {sp.diff(residual, y2)}")


def compute_line_vertical_derivatives():
    x1, x2 = sp.symbols("x1 x2")
    residual = x1 - x2

    print("Residual: R = x1 - x2")
    print(f"∂R/∂x1 = {sp.diff(residual, x1)}")
    print(f"∂R/∂x2 = {sp.diff(residual, x2)}")


def compute_lines_parallel_derivatives():
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")  # Line 1
    x3, y3, x4, y4 = sp.symbols("x3 y3 x4 y4")  # Line 2

    v1_x, v1_y = x2 - x1, y2 - y1
    v2_x, v2_y = x4 - x3, y4 - y3

    # Lines are parallel when cross product is zero.
    residual = v1_x * v2_y - v1_y * v2_x
    variables = [x1, y1, x2, y2, x3, y3, x4, y4]

    print("Residual: R = (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)")

    for var in variables:
        deriv = sp.diff(residual, var)
        simplified_deriv = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified_deriv}")


def compute_lines_perpendicular_derivatives():
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")  # Line 1
    x3, y3, x4, y4 = sp.symbols("x3 y3 x4 y4")  # Line 2

    v1_x, v1_y = x2 - x1, y2 - y1
    v2_x, v2_y = x4 - x3, y4 - y3

    # Lines are perpendicular when dot product is zero.
    residual = v1_x * v2_x + v1_y * v2_y
    variables = [x1, y1, x2, y2, x3, y3, x4, y4]

    print("Residual: R = (x2-x1)*(x4-x3) + (y2-y1)*(y4-y3)")

    for var in variables:
        deriv = sp.diff(residual, var)
        simplified_deriv = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified_deriv}")


def compute_lines_equal_length_derivatives():
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")  # Line 1
    x3, y3, x4, y4 = sp.symbols("x3 y3 x4 y4")  # Line 2

    len1_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    len2_squared = (x4 - x3) ** 2 + (y4 - y3) ** 2
    len1 = sp.sqrt(len1_squared)
    len2 = sp.sqrt(len2_squared)
    residual = len1 - len2

    variables = [x1, y1, x2, y2, x3, y3, x4, y4]

    print("Residual: R = |L1| - |L2|")

    for var in variables:
        deriv = sp.diff(residual, var)
        simplified_deriv = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified_deriv}")


def compute_line_line_angle_derivatives():
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")  # Line 1
    x3, y3, x4, y4 = sp.symbols("x3 y3 x4 y4")  # Line 2
    alpha = sp.Symbol("alpha")

    v1 = sp.Matrix([x2 - x1, y2 - y1])
    v2 = sp.Matrix([x4 - x3, y4 - y3])

    # Get the elements as scalars rather than matrix elements
    v1_x, v1_y = v1[0], v1[1]
    v2_x, v2_y = v2[0], v2[1]

    # Cross and dot products
    cross_2d = v1_x * v2_y - v1_y * v2_x
    dot_prod = v1.dot(v2)

    # Single residual using atan2
    residual = sp.atan2(cross_2d, dot_prod) - alpha

    variables = [x1, y1, x2, y2, x3, y3, x4, y4]

    print("Residual: R = atan2(v1×v2, v1·v2) - α")

    for var in variables:
        deriv = sp.diff(residual, var)
        simplified_deriv = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified_deriv}")


def compute_line_line_distance_derivatives():
    x1, y1, x2, y2 = sp.symbols("x1 y1 x2 y2")  # Line 1: p1 to p2
    xp, yp = sp.symbols("xp yp")  # Point on Line 2
    d = sp.Symbol("d")

    # Use signed distance (no absolute value)
    v_x, v_y = x2 - x1, y2 - y1
    w_x, w_y = xp - x1, yp - y1

    # Signed cross product
    cross_2d = v_x * w_y - v_y * w_x

    # Line magnitude
    v_mag = sp.sqrt(v_x**2 + v_y**2)

    # Signed distance
    signed_distance = cross_2d / v_mag

    # Two possible residual formulations:
    residual_signed = signed_distance - d  # d can be positive or negative
    # residual_squared = signed_distance**2 - d**2  # Always positive constraint

    variables = [x1, y1, x2, y2, xp, yp]

    print("Residual: R = (|v × w| / |v|) - d")

    for var in variables:
        deriv = sp.diff(residual_signed, var)
        simplified = sp.simplify(deriv)
        print(f"∂R/∂{var} = {simplified}")


if __name__ == "__main__":
    # compute_point_fixed_derivatives()
    # compute_point_point_euclidean_distance_derivatives()
    # compute_point_point_x_distance_derivatives()
    # compute_point_point_y_distance_derivatives()
    # compute_line_horizontal_derivatives()
    # compute_line_vertical_derivatives()
    # compute_lines_parallel_derivatives()
    # compute_lines_perpendicular_derivatives()
    # compute_lines_equal_length_derivatives()
    # compute_line_line_angle_derivatives()
    compute_line_line_distance_derivatives()
