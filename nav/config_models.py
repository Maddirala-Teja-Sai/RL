import numpy as np
import random
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

PI = np.pi

"""
This file contains all the different config settings used to communicate between different modules of this repo
"""

# A basic vector class, also used to represent points
class Vector2(BaseModel):
    x: float
    y: float

    def to_numpy(self):
        return np.array([self.x, self.y])


class Line(BaseModel):
    start: Vector2
    end: Vector2


class Rectangle(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    center: Vector2
    width: float
    height: float
    rotation: float = 0  # Rotation in degrees


class AgentConfig(BaseModel):
    start_pos: Rectangle # Exact point will be sampled
    goal_pos: Rectangle
    radius: float = 0.02
    max_speed: float
    agent_col: str = "blue"
    max_range: float = 0.25
    fov_degrees: float = 210.0


class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Vector2
    radius: float


class ObstacleSchedule(BaseModel):
    speed: Optional[float] = None
    direction: Optional[Vector2] = None
    angular_speed: Optional[float] = None # degrees
    rotating_up: Optional[bool] = None
    boundary_x_min: Optional[float] = None
    boundary_x_max: Optional[float] = None


class ObstacleConfig(BaseModel):
    shape: Union[Rectangle, Circle]
    schedule: Optional[ObstacleSchedule] = None
    noise: Optional[float] = None


class PolygonBoundaryConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    vertices: List[Vector2]


class SwitchConfig(BaseModel):
    zone: Union[Rectangle, Circle]
    reward: float = 0.25
    color: str = "cyan"


class GateConfig(BaseModel):
    shape: Rectangle
    opens_when_active_switches: int = 1
    trigger_zone: Optional[Union[Rectangle, Circle]] = None  # Gate opens when agent enters this zone


class EnvConfig(BaseModel):
    boundary: PolygonBoundaryConfig
    obstacles: List[ObstacleConfig] = []
    agents: List[AgentConfig]
    max_time: int
    num_rays: int = 60
    goal_threshold: float = 0.02
    repeat_steps: int = 2
    num_agents_per_group: int = 1
    state_image_size: int = 64
    switches: List[SwitchConfig] = []
    gates: List[GateConfig] = []
    simultaneous_arrival_bonus: float = 0.0
    simultaneous_arrival_window: int = 50
    formation_reward_weight: float = 0.0
    use_shared_goals: bool = False
    terminal_strategy: Literal["any", "all", "individual"] = "any"
    gate_open_bonus: float = 5.0          # Bonus when a gate opens
    potential_shaping_scale: float = 5.0  # Scale of potential-based goal shaping
    stuck_zone_penalty: float = 80.0      # Penalty for re-entering a stuck zone
    winner_team_bonus: float = 0.0        # Bonus for teammates of the winning agent
    loser_team_penalty: float = 0.0       # Penalty for agents on the losing team
    collision_penalty: float = 10.0       # Penalty for hitting walls/obstacles
    agent_collision_penalty: float = 2.0  # Penalty for bumping into teammates
    not_reached_goal_penalty: float = 0.0 # Penalty for agents that haven't reached goal when episode ends
    goal_reach_bonus: float = 100.0       # Bonus for reaching the goal
