from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Set

from newton import backend as nb


class Primitive(ABC):
    def __init__(self, id: str):
        self.id = id

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
        # State is the center coordinates and radius.
        var_ids = self.get_variable_ids()
        center_x = variable_values[var_ids[0]]
        center_y = variable_values[var_ids[1]]
        radius = variable_values[var_ids[2]]

        return nb.np.array([center_x, center_y, radius])

    def get_variable_ids(self) -> List[str]:
        return [f"{self.center.id}_x", f"{self.center.id}_y", f"{self.id}_radius"]

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id, self.center.id}
