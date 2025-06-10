from marshmallow_dataclass import dataclass

from robot_utils.py.enum import StrEnum
from robot_utils.serialize.dataclass import AbstractDataclass, default_field


class JointType(StrEnum):
    Prismatic = "Prismatic"
    Revolute = "Revolute"
    Screw = "Screw"
    Planar = "Planar"
    Ball = "Ball"


@dataclass
class ExpectedScore(AbstractDataclass):
    col: float = 0
    cop: float = 0
    radius: float = 0
    pitch: float = 0


class JointExpectedScores(AbstractDataclass):
    prismatic: ExpectedScore = default_field(ExpectedScore())
    revolute: ExpectedScore = default_field(ExpectedScore())
    planar: ExpectedScore = default_field(ExpectedScore())
    screw: ExpectedScore = default_field(ExpectedScore())
    ball: ExpectedScore = default_field(ExpectedScore())
