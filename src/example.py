import logging
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc
from matplotlib.patches import Circle as CirclePatch
from pyinstrument import Profiler

from newton.constants import CONFIG_USE_SPARSE_SOLVE
from newton.constraints import (
    ArcRadius,
    CircleRadius,
    LineHorizontal,
    LineLineDistance,
    LinesEqualLength,
    LinesParallel,
    LinesPerpendicular,
    LineTangentToCircle,
    LineVertical,
    PointFixed,
    PointPointCoincident,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import configure_logging, logger
from newton.primitives import Circle, CircularArc, Line, Point, Primitive
from newton.solver_dense import Solver2DDense
from newton.solver_sparse import Solver2DSparse

configure_logging(level=logging.DEBUG)

PLOT = True


def add_random_error(points: list[Point], error_range: float = 1.0, seed: int = 42):
    """
    Add random error to point coordinates in the specified range.

    Args:
        points: List of points to modify
        error_range: Maximum error magnitude (default: 1.0)
        seed: Random seed for reproducibility

    Returns:
        The modified points (same objects, modified in-place)
    """
    rng = random.Random(seed)
    for point in points:
        # Add random error in the range [-error_range, error_range]
        point.x += rng.uniform(-error_range, error_range)
        point.y += rng.uniform(-error_range, error_range)
    return points


def draw_point(point: Point, color: str, prime: bool = False):
    plt.plot(point.x, point.y, marker="o", color=color, markersize=5, linestyle="None")
    id_label = f"{point.id}'" if prime else point.id
    plt.text(point.x, point.y + 0.1, id_label, fontsize=9, ha="center", va="bottom")


def draw_line(line: Line, color: str, label: str | None = None):
    plt.plot(
        [line.p1.x, line.p2.x],
        [line.p1.y, line.p2.y],
        color=color,
        label=label,
        linewidth=1.5,
    )


def draw_circle(circle: Circle, color: str, label: str | None = None):
    center_pos = (circle.center.x, circle.center.y)
    radius = circle.radius

    circle_patch = CirclePatch(
        center_pos, radius, color=color, fill=False, label=label, linewidth=1.5
    )
    ax = plt.gca()
    ax.add_patch(circle_patch)


def draw_arc(arc: CircularArc, color: str, label: str | None = None):
    center_pos = (arc.center.x, arc.center.y)

    # Calculate radius from center to start point
    radius = np.sqrt(
        (arc.start.x - arc.center.x) ** 2 + (arc.start.y - arc.center.y) ** 2
    )

    # Calculate angles in degrees
    start_angle = np.degrees(
        np.arctan2(arc.start.y - arc.center.y, arc.start.x - arc.center.x)
    )
    end_angle = np.degrees(
        np.arctan2(arc.end.y - arc.center.y, arc.end.x - arc.center.x)
    )

    # Calculate arc span
    angle_span = end_angle - start_angle
    if angle_span < 0:
        angle_span += 360

    arc_patch = Arc(
        center_pos,
        2 * radius,
        2 * radius,  # width, height
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=1.5,
        label=label,
    )

    ax = plt.gca()
    ax.add_patch(arc_patch)


def plot_geometry(
    primitives: list[Primitive], color: str, label: str, prime: bool = False
):
    has_labeled_line = False
    has_labeled_circle = False

    for prim in primitives:
        if isinstance(prim, Line):
            line_label = label if not has_labeled_line else None
            draw_line(prim, color=color, label=line_label)
            has_labeled_line = True
        elif isinstance(prim, Circle):
            circle_label = label if not has_labeled_circle else None
            draw_circle(prim, color=color, label=circle_label)
            has_labeled_circle = True
        elif isinstance(prim, CircularArc):
            draw_arc(prim, color=color, label=label)
        elif isinstance(prim, Point):
            # Points are drawn by themselves after lines/circles to be on top.
            pass

    # Draw all points last so they appear on top of lines/circles.
    for prim in primitives:
        if isinstance(prim, Point):
            draw_point(prim, color=color, prime=prime)


def constrain_rectangles():
    # First square
    p1 = Point(x=1.0, y=1.0, id="P1")
    p2 = Point(x=4.5, y=1.5, id="P2")
    p3 = Point(x=4.0, y=3.5, id="P3")
    p4 = Point(x=1.5, y=3.0, id="P4")
    points1 = [p1, p2, p3, p4]

    l_bottom1 = Line(p1, p2, "L_Bottom1")
    l_right1 = Line(p2, p3, "L_Right1")
    l_top1 = Line(p3, p4, "L_Top1")
    l_left1 = Line(p4, p1, "L_Left1")
    lines1 = [l_bottom1, l_right1, l_top1, l_left1]

    constraints1 = [
        PointFixed(point=p1),
        LineHorizontal(line=l_bottom1),
        LineHorizontal(line=l_top1),
        LineVertical(line=l_left1),
        LineVertical(line=l_right1),
        PointPointEuclideanDistance(p1, p2, distance=4.0),
        PointPointEuclideanDistance(p1, p4, distance=3.0),
    ]

    # Second square
    p5 = Point(x=2.0, y=2.0, id="P5")
    p6 = Point(x=5.5, y=3.5, id="P6")
    p7 = Point(x=5.0, y=4.5, id="P7")
    p8 = Point(x=2.5, y=4.0, id="P8")
    points2 = [p5, p6, p7, p8]

    l_bottom2 = Line(p5, p6, "L_Bottom2")
    l_right2 = Line(p6, p7, "L_Right2")
    l_top2 = Line(p7, p8, "L_Top2")
    l_left2 = Line(p8, p5, "L_Left2")
    lines2 = [l_bottom2, l_right2, l_top2, l_left2]

    constraints2 = [
        PointFixed(point=p5),
        LineHorizontal(line=l_bottom2),
        LineHorizontal(line=l_top2),
        LineVertical(line=l_left2),
        LineVertical(line=l_right2),
        PointPointEuclideanDistance(p5, p6, distance=4.0),
        PointPointEuclideanDistance(p5, p8, distance=4.0),
        # Add a duplicate constraint to test conflict detection.
        # PointPointEuclideanDistance(p5, p6, distance=4.0),  # Duplicate
        # Add a conflicting constraint to test conflict detection.
        # LineHorizontal(line=l_right2),  # This should conflict with the vertical line.
    ]

    # Combine all geometry and constraints for the solver
    all_points: list[Primitive] = []
    all_points.extend(points1)
    all_points.extend(points2)
    all_lines = lines1 + lines2
    all_constraints = constraints1 + constraints2

    # Plot initial state
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(all_points + all_lines, color="red", label="Initial")  # type: ignore

    # Sooooooolve it.
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(all_points, all_constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(all_points + all_lines, color="blue", label="Solved", prime=True)  # type: ignore

        plt.legend()
        plt.title("Rectangles From Constraints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_parallel_offset():
    p0 = Point(x=1.0, y=1.0, id="P0")
    p1 = Point(x=2.5, y=1.5, id="P1")
    p2 = Point(x=-1.5, y=3.0, id="P2")
    p3 = Point(x=3.5, y=4.0, id="P3")

    line1 = Line(p0, p1, "L1")
    line2 = Line(p2, p3, "L2")

    points: list[Primitive] = []
    points = [p0, p1, p2, p3]
    lines = [line1, line2]

    # Define the constraints in a specific order to show the dependency.
    constraints = [
        PointFixed(point=p0),
        PointPointXDistance(p0, p1, distance=4),
        PointPointYDistance(p0, p1, distance=9),
        LinesParallel(line1, line2),
        LinesEqualLength(line1, line2),
        LineLineDistance(line1, line2, distance=2.0),
        PointPointXDistance(p2, p0, distance=1),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(points + lines, color="red", label="Initial")  # type: ignore

    # Sooooooolve it.
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(points, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(points + lines, color="blue", label="Solved", prime=True)  # type: ignore

        plt.legend()
        plt.title("Parallel Offset Lines")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_underdetermined():
    """
    We want say three lines, two fully defined, one not quite.
    """
    p0 = Point(x=0.0, y=0.0, id="P0")
    p1 = Point(x=4.0, y=4.0, id="P1")
    p2 = Point(x=4.0, y=4.0, id="P2")
    p3 = Point(x=8.0, y=6.0, id="P3")
    p4 = Point(x=8.0, y=6.0, id="P4")
    p5 = Point(x=12.0, y=6.0, id="P5")  # This point is not fully constrained.

    # Modify our points a bit and then we'll pull these back with constraints.
    points_to_modify = [p1, p2, p3, p4, p5]
    add_random_error(points_to_modify, error_range=1.0, seed=42)

    points: list[Primitive] = []
    points = [p0, *points_to_modify]

    line1 = Line(p0, p1, "L1")
    line2 = Line(p2, p3, "L2")
    line3 = Line(p4, p5, "L3")  # This line is not fully constrained.

    lines = [line1, line2, line3]

    # Define the constraints in a specific order to show the dependency.
    constraints = [
        PointFixed(point=p0),
        PointPointXDistance(p0, p1, distance=4),
        PointPointYDistance(p0, p1, distance=4),
        PointPointEuclideanDistance(p1, p2, distance=0),  # Coincident.
        PointPointXDistance(p2, p3, distance=4),
        PointPointYDistance(p2, p3, distance=2),
        PointPointEuclideanDistance(p3, p4, distance=0),  # Coincident.
        PointPointXDistance(p4, p5, distance=4),
        # This would make it fully constrained.
        # PointPointYDistance(p4, p5, distance=0),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(points + lines, color="red", label="Initial")  # type: ignore
        # plt.show()

    # Sooooooolve it.
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(points, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(points + lines, color="blue", label="Solved", prime=True)  # type: ignore

        plt.legend()
        plt.title("Underconstrained System")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_simple_circle():
    center_point = Point(x=1.0, y=2.0, id="C1_P")

    # The initial radius is 5.0, but we will constrain it to 3.0.
    circle = Circle(center=center_point, radius=5.0, id="C1")

    # The solver operates on a single list of all primitives.
    # The Circle primitive adds "C1_radius" as a variable.
    # The Point primitive adds "C1_P_x" and "C1_P_y" as variables.
    all_primitives = [center_point, circle]

    # Define the constraints
    constraints = [
        PointFixed(point=center_point),
        CircleRadius(circle=circle, radius=3.0),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(all_primitives, color="red", label="Initial", prime=False)

    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(all_primitives, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(all_primitives, color="blue", label="Solved", prime=True)
        plt.legend()
        plt.title("Simple Circle Constraint")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_tangent_circle_to_line():
    # Define the primitives.
    p0 = Point(x=0.0, y=0.0, id="P0")
    p1 = Point(x=1.0, y=2.0, id="P1")
    p2 = Point(x=5.0, y=1.0, id="P2")
    line = Line(p1, p2, "L1")

    # Start the circle away from the line.
    center = Point(x=3.0, y=5.0, id="C1_P")
    circle = Circle(center=center, radius=1.0, id="C1")

    all_primitives = [p0, p1, p2, line, center, circle]

    # Define the constraints.
    constraints = [
        PointFixed(point=p0),
        PointPointCoincident(p0, p1),
        LineHorizontal(line=line),
        CircleRadius(circle=circle, radius=2.0),
        PointPointXDistance(p1, center, distance=2.0),
        LineTangentToCircle(line=line, circle=circle),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(all_primitives, color="red", label="Initial")

    # Solve.
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(all_primitives, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(all_primitives, color="blue", label="Solved", prime=True)
        plt.legend()
        plt.title("Line Tangent to Circle")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_simple_arc():
    center = Point(x=0.0, y=0.0, id="A1_C")
    start = Point(x=0.0, y=3.0, id="A1_S")
    end = Point(x=0.0, y=-3.0, id="A1_E")

    arc = CircularArc(center=center, start=start, end=end, id="A1")

    all_primitives = [center, start, end, arc]

    # Define the constraints that will force the arc to change.
    r = 2.0
    constraints = [
        PointFixed(point=center),
        ArcRadius(arc=arc, radius=r),
        PointPointXDistance(center, start, distance=0.0),
        PointPointXDistance(center, end, distance=1.0),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(all_primitives, color="red", label="Initial", prime=False)

    # Pass them to the solver. The solver handles the rest automatically!
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(all_primitives, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(all_primitives, color="blue", label="Solved", prime=True)

        plt.legend()
        plt.title("Arc Constraint")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.show()


def test_determinism():
    n_passed = 0
    n_failed = 0
    for i in range(100):
        try:
            constrain_underdetermined()
            n_passed += 1
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")
            n_failed += 1

    print(f"Determinism test completed: {n_passed} passed, {n_failed} failed.")


def constrain_perpendicular_with_shared_vertex():
    """
    Build a simple polyline p1-p2-p3 with two segments:
    L1 = (p1, p2), L2 = (p2, p3)

    Then constrain L1 perp L2 at the shared vertex p2.
    """
    p1 = Point(x=0.0, y=0.0, id="P1")
    p2 = Point(x=3.2, y=0.7, id="P2")
    p3 = Point(x=4.8, y=2.1, id="P3")

    l1 = Line(p1, p2, id="L1")
    l2 = Line(p2, p3, id="L2")

    # Ensure segments share the middle vertex.
    assert l1.p2 is l2.p1, "Segments must share the middle vertex"

    primitives = [p1, p2, p3, l1, l2]

    constraints = [
        PointFixed(point=p1),
        LineHorizontal(l1),
        PointPointXDistance(p1, p2, distance=4.0),
        PointPointYDistance(p2, p3, distance=3.0),
        LinesPerpendicular(l1, l2),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(primitives, color="red", label="Initial", prime=False)

    # Solve.
    Solver2D = Solver2DSparse if CONFIG_USE_SPARSE_SOLVE else Solver2DDense
    solver = Solver2D(primitives=primitives, constraints=constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(primitives, color="blue", label="Solved", prime=True)
        plt.legend()
        plt.title("Perpendicular Lines with Shared Vertex")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    constrain_rectangles()
    # constrain_parallel_offset()
    # constrain_underdetermined()
    # constrain_simple_circle()
    # constrain_tangent_circle_to_line()
    # constrain_simple_arc()
    # constrain_perpendicular_with_shared_vertex()

    profiler.stop()
    profiler.print()

    # test_determinism()
