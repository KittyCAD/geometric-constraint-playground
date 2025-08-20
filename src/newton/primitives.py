from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, ForwardRef, List, Mapping, Set

from newton import backend as nb

if TYPE_CHECKING:
    from newton.constraints import Constraint
else:
    # Create a forward reference for runtime
    Constraint = ForwardRef("Constraint")


class Primitive(ABC):
    def __init__(self, id: str):
        self.id = id

    def build_definitional_constraints(self) -> List[Constraint]:
        # Build the constraints that define this primitive; generally there aren't any.
        return []

    @abstractmethod
    def get_initial_variable_values(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_state(self, variable_values: Mapping[str, float]) -> nb.Vector:
        pass

    @abstractmethod
    def get_variable_ids(self) -> List[str]:
        pass

    @abstractmethod
    def get_involved_primitive_ids(self) -> Set[str]:
        pass


@dataclass
class Point(Primitive):
    x: float
    y: float
    id: str

    def __post_init__(self):
        super().__init__(self.id)

    def get_initial_variable_values(self) -> Dict[str, float]:
        return {
            f"{self.id}_x": self.x,
            f"{self.id}_y": self.y,
        }

    def get_state(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # State is (x, y) coordinates.
        var_ids = self.get_variable_ids()
        x = variable_values[var_ids[0]]
        y = variable_values[var_ids[1]]

        return nb.np.array([x, y])

    def get_variable_ids(self) -> List[str]:
        return [f"{self.id}_x", f"{self.id}_y"]

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id}


@dataclass
class Line(Primitive):
    p1: Point
    p2: Point
    id: str

    def __post_init__(self):
        super().__init__(self.id)

    def get_initial_variable_values(self) -> Dict[str, float]:
        # A Line owns no variables.
        return {}

    def get_state(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # A line has no independent state, only that of its points.
        return nb.np.array([])

    def get_variable_ids(self) -> List[str]:
        # A line itself introduces no new variables to the system.
        # The variables of its points will be discovered when the points
        # themselves are processed by the solver.
        return []

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id, self.p1.id, self.p2.id}


@dataclass
class Circle(Primitive):
    center: Point
    radius: float
    id: str

    def __post_init__(self):
        super().__init__(self.id)

    def get_initial_variable_values(self) -> Dict[str, float]:
        # A Circle only owns its radius variable. Its centre Point will handle its own.
        return {f"{self.id}_radius": self.radius}

    def get_state(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # Get centre from the centre point, radius from our own variable.
        p_center = self.center.get_state(variable_values)
        radius_var_id = f"{self.id}_radius"
        radius = variable_values[radius_var_id]

        return nb.np.array([p_center[0], p_center[1], radius])

    def get_variable_ids(self) -> List[str]:
        return [f"{self.id}_radius"]

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id, self.center.id}


@dataclass
class CircularArc(Primitive):
    # Circular arc defined by centre point and two endpoint points.
    center: Point
    start: Point
    end: Point
    id: str

    def __post_init__(self):
        super().__init__(self.id)

    def build_definitional_constraints(self) -> List[Constraint]:
        # A circular arc is defined by its centre and its endpoints, but we need those
        # endpoints to be the same distance from the centre, i.e., constant radius.
        from newton.constraints import PointsEquidistant

        return [PointsEquidistant(center=self.center, p1=self.start, p2=self.end)]

    def get_initial_variable_values(self) -> Dict[str, float]:
        # A CircularArc owns no variables, only its constituent points do.
        return {}

    def get_state(self, variable_values: Mapping[str, float]) -> nb.Vector:
        # A circular arc has no independent state, only that of its points.
        return nb.np.array([])

    def get_variable_ids(self) -> List[str]:
        # A circular arc introduces no new variables to the system, just use the points.
        return []

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id, self.center.id, self.start.id, self.end.id}
