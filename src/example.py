import logging
import random

import matplotlib.pyplot as plt
from pyinstrument import Profiler

from newton.constraints import (
    LineHorizontal,
    LineLineDistance,
    LinesEqualLength,
    LinesParallel,
    LineVertical,
    PointFixed,
    PointPointEuclideanDistance,
    PointPointXDistance,
    PointPointYDistance,
)
from newton.logging_config import configure_logging
from newton.primitives import Line, Point
from newton.solver_dense import Solver2DDense
from newton.solver_sparse import Solver2DSparse

configure_logging(level=logging.INFO)

USE_SPARSE = True
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


def plot_geometry(
    points: list[Point], lines: list[Line], color: str, label: str, prime: bool = False
):
    for i, line in enumerate(lines):
        line_label = label if i == 0 else None
        draw_line(line, color=color, label=line_label)

    for point in points:
        draw_point(point, color=color, prime=prime)


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
    all_points = points1 + points2
    all_lines = lines1 + lines2
    all_constraints = constraints1 + constraints2

    # Plot initial state
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(all_points, all_lines, color="red", label="Initial")

    # Sooooooolve it.
    Solver2D = Solver2DSparse if USE_SPARSE else Solver2DDense
    solver = Solver2D(all_points, all_constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(all_points, all_lines, color="blue", label="Solved", prime=True)

        plt.legend()
        plt.title("Rectangles From Constraints")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def constrain_decomposable():
    """
    Creates a system that is fully connected, but can be sequentially decomposed
    by the StructuralAnalyzer.
    """
    p0 = Point(x=1.0, y=1.0, id="P0")
    p1 = Point(x=2.5, y=1.5, id="P1")
    p2 = Point(x=1.5, y=3.0, id="P2")
    p3 = Point(x=3.5, y=4.0, id="P3")

    line1 = Line(p0, p1, "L1")
    line2 = Line(p2, p3, "L2")

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
        PointPointXDistance(p0, p2, distance=1),
    ]

    # Plot initial state.
    if PLOT:
        plt.figure(figsize=(8, 8))
        plot_geometry(points, lines, color="red", label="Initial")

    # Sooooooolve it.
    Solver2D = Solver2DSparse if USE_SPARSE else Solver2DDense
    solver = Solver2D(points, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(points, lines, color="blue", label="Solved", prime=True)

        plt.legend()
        plt.title("Decomposable System Solved Sequentially")
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
        plot_geometry(points, lines, color="red", label="Initial")
        # plt.show()

    # Sooooooolve it.
    Solver2D = Solver2DSparse if USE_SPARSE else Solver2DDense
    solver = Solver2D(points, constraints)
    solver.solve()

    # Plot final state.
    if PLOT:
        plot_geometry(points, lines, color="blue", label="Solved", prime=True)

        plt.legend()
        plt.title("Underconstrained System")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    # constrain_rectangles()
    # constrain_decomposable()
    # constrain_underdetermined()

    profiler.stop()
    profiler.print()
