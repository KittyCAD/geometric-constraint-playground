from dataclasses import dataclass, field
from typing import List


@dataclass
class Point:
    """
    Represents a 2D point with an initial position.
    """

    x: float
    y: float
    id: str = ""
    constraints: List = field(default_factory=list, repr=False)


@dataclass
class Line:
    """
    Represents a line segment defined by two points.
    """

    p1: Point
    p2: Point
    id: str = ""
