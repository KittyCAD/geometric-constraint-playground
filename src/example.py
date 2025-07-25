import matplotlib.pyplot as plt

from newton.constraints import (
    LineHorizontal,
    LineVertical,
    PointFixed,
    PointPointDistance,
)
from newton.primitives import Line, Point
from newton.solver import Solver2D

if __name__ == "__main__":
    # First square.
    p1 = Point(x=1.0, y=1.0, id="P1")
    p2 = Point(x=4.5, y=1.5, id="P2")
    p3 = Point(x=4.0, y=3.5, id="P3")
    p4 = Point(x=1.5, y=3.0, id="P4")
    points = [p1, p2, p3, p4]

    # Define lines connecting the points.
    l_bottom = Line(p1, p2, "L_Bottom")
    l_right = Line(p2, p3, "L_Right")
    l_top = Line(p3, p4, "L_Top")
    l_left = Line(p4, p1, "L_Left")
    lines = [l_bottom, l_right, l_top, l_left]

    constraints = [
        #  Anchor one corner.
        PointFixed(point=p1),
        # Make the top and bottom sides horizontal.
        LineHorizontal(line=l_bottom),
        LineHorizontal(line=l_top),
        # Make the left and right sides vertical.
        LineVertical(line=l_left),
        LineVertical(line=l_right),
        # Define the width and height of the box.
        PointPointDistance(p1, p2, distance=4.0),
        PointPointDistance(p1, p4, distance=3.0),
    ]

    # Define a second square to check if we can identify it.
    p5 = Point(x=2.0, y=2.0, id="P5")
    p6 = Point(x=5.5, y=2.5, id="P6")
    p7 = Point(x=5.0, y=4.5, id="P7")
    p8 = Point(x=2.5, y=4.0, id="P8")
    points.extend([p5, p6, p7, p8])

    # Define lines connecting the second square points.
    l_bottom2 = Line(p5, p6, "L_Bottom2")
    l_right2 = Line(p6, p7, "L_Right2")
    l_top2 = Line(p7, p8, "L_Top2")
    l_left2 = Line(p8, p5, "L_Left2")
    lines.extend([l_bottom2, l_right2, l_top2, l_left2])

    constraints.extend(
        [
            # Anchor one corner of the second square.
            PointFixed(point=p5),
            # Make the top and bottom sides horizontal.
            LineHorizontal(line=l_bottom2),
            LineHorizontal(line=l_top2),
            # Make the left and right sides vertical.
            LineVertical(line=l_left2),
            LineVertical(line=l_right2),
            # Define the width and height of the second box.
            PointPointDistance(p5, p6, distance=4.0),
            PointPointDistance(p5, p8, distance=3.0),
        ]
    )

    def plot_shape(points, color, label, prime=False):
        plot_points = points + [points[0]]
        xs = [p.x for p in plot_points]
        ys = [p.y for p in plot_points]
        plt.plot(xs, ys, marker="o", color=color, label=label)

        for p in points:
            id = p.id if not prime else f"{p.id}'"
            plt.text(p.x, p.y + 0.1, id, fontsize=9, ha="center", va="bottom")

    # Plot initial state.
    plt.figure(figsize=(8, 8))
    plot_shape(points, color="red", label="Initial")

    # Sooooooolve it.
    solver = Solver2D(points, constraints)
    solver.solve()

    # Plot final state.
    plot_shape(points, color="blue", label="Solved", prime=True)

    plt.legend()
    plt.title("Rectangle From Constraints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
