from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set


class Primitive(ABC):
    def __init__(self, id: str):
        self.id = id

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

    def get_variable_ids(self) -> List[str]:
        return []

    def get_involved_primitive_ids(self) -> Set[str]:
        return {self.id, self.p1.id, self.p2.id}
