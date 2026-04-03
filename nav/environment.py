import pettingzoo
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .obstacles import PolygonBoundary
from .utils import sample_point_in_rectangle, convert_to_polar
from .config_models import *
from .renderer_models import RenderState, AgentState, BoundaryState, SwitchState
from .obstacles import ObstacleFactory
import yaml
from typing import Dict, List, Any, Literal, Optional
from pydantic import BaseModel
from .ray_intersection import (
    batch_ray_intersection_detailed,
    create_lidar_rays,
    RayIntersectionOutput,
    NoHit,
)
from collections import deque
from .live_renderer import SimulationWindow

DELTA_T = 1 / 60

COLLIDING_WITH_TYPES = Literal["obstacle", "boundary", "agent"]


class CollisionData(BaseModel):
    is_colliding: bool = False
    colliding_with: Optional[COLLIDING_WITH_TYPES] = None


class Agent:
    def __init__(
        self,
        agent_config: AgentConfig,
        goal_threshold: float = 0.02,
    ):
        self.config = agent_config
        self.pos = self.config.start_pos.center.to_numpy()
        self.start_pos = self.pos.copy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_sample_area = self.config.goal_pos
        self.goal_pos = self.config.goal_pos.center.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 10
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False
        self.old_pos = self.pos.copy()
        self.goal_threshold = goal_threshold
        self.last_reward = 0

    def has_reached_goal(self):
        return np.linalg.norm(self.goal_pos - self.pos) < (
            self.goal_threshold + self.radius
        )

    def get_state_dict(self):
        original_distance_to_goal = np.linalg.norm(self.goal_pos - self.start_pos)
        current_distance_to_goal = np.linalg.norm(self.goal_pos - self.pos)
        progress = (
            original_distance_to_goal - current_distance_to_goal
        ) / (original_distance_to_goal + 1e-10)

        goal_vector = self.goal_pos - self.pos
        dist_norm = np.linalg.norm(goal_vector)
        if dist_norm > 1e-10:
            goal_vector = goal_vector / dist_norm
        
        cosine_angle = goal_vector.dot(self.direction)
        speed_ratio = self.current_speed / self.config.max_speed

        return {
            "state_vector": [
                progress,  # progress towards goal 0-1
                cosine_angle,  # cosine of angle between goal vector and direction vector
                speed_ratio,  # ratio of current speed to max speed
                speed_ratio * cosine_angle,
                current_distance_to_goal,
                goal_vector[0],
                goal_vector[1],
            ],
        }

    def get_action(self, lidar_observation: np.ndarray):
        self.lidar_observation_history.append(lidar_observation)
        return None  # TODO: Implement action selection

    def update_pos(self, delta_t: float = 1 / 30):
        if self.goal_reached:
            return
        if not self.active:
            return
        self.old_pos = self.pos.copy()
        self.pos = self.pos + (self.direction * self.current_speed * delta_t)
        self.goal_reached = self.has_reached_goal()

    def convert_velocity_to_global(self, velocity: Vector2, heading_vector: Vector2):
        dist = np.linalg.norm(heading_vector)
        if dist > 1e-10:
            u_x = heading_vector[0] / dist
            u_y = heading_vector[1] / dist
            a_x = velocity.x
            a_y = velocity.y
            global_vx = a_x * u_y + a_y * u_x
            global_vy = -a_x * u_x + a_y * u_y
            target_velocity_global = np.array([global_vx, global_vy])
        else:
            target_velocity_global = velocity.to_numpy()
        return target_velocity_global

    def apply_target_velocity(self, target_velocity: Vector2):
        if not self.active:
            self.current_speed = 0
            return

        target_velocity_global = self.convert_velocity_to_global(
            target_velocity, (self.goal_pos - self.pos)
        )

        current_velocity = self.current_speed * self.direction
        force = target_velocity_global - current_velocity
        new_velocity = current_velocity + force * (self.response_time * DELTA_T)

        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > 1e-10:
            self.current_speed = velocity_magnitude
            self.direction = new_velocity / velocity_magnitude
        else:
            self.current_speed = 0.0

        self.current_speed = min(self.current_speed, self.config.max_speed)


class Environment(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "name": "navigation_v0"}

    def __init__(
        self,
        config: dict[str, Any] | EnvConfig,
        render_mode: Optional[str] = None,
        avoid_collision_checks: bool = False,
    ):
        if isinstance(config, dict):
            config = EnvConfig(**config)

        self.config = config
        self.state_image_size = config.state_image_size
        self.boundary = PolygonBoundary(config.boundary)
        self.num_groups = len(config.agents)
        self.avoid_collision_checks = avoid_collision_checks

        agent_configs = self.preprocess_agent_configs(
            config.agents, config.num_agents_per_group
        )

        self.agents_dict: Dict[str, Agent] = {
            f"agent_{i}": Agent(
                agent_config,
                goal_threshold=self.config.goal_threshold,
            )
            for i, agent_config in enumerate(agent_configs)
        }
        for agent_id, agent in self.agents_dict.items():
            agent.agent_id = agent_id

        self.n_agents = len(self.agents_dict)
        self.agent_group_map = {
            f"agent_{idx}": idx // max(1, config.num_agents_per_group)
            for idx in range(self.n_agents)
        }
        self.group_members = {
            group_idx: [
                agent_id for agent_id, mapped_group in self.agent_group_map.items()
                if mapped_group == group_idx
            ]
            for group_idx in range(self.num_groups)
        }
        self.state_dim = len(
            next(iter(self.agents_dict.values())).get_state_dict()["state_vector"]
        )
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]
        self.switches = config.switches
        self.gates = config.gates
        self.gate_obstacles = [
            ObstacleFactory.create(
                ObstacleConfig(shape=gate.shape, schedule=None, noise=0.0)
            )
            for gate in self.gates
        ]
        self.active_switches: set[int] = set()
        self.gate_open_mask = [False for _ in self.gates]
        self.gate_just_opened = False
        self.group_goal_steps = {agent_id: None for agent_id in self.agents_dict}
        self.group_bonus_awarded = {group_idx: False for group_idx in self.group_members}
        self.pending_group_bonus = {agent_id: 0.0 for agent_id in self.agents_dict}
        self.num_steps = 0

        self.goal_pool = []
        self.occupied_goals = set()
        if self.config.use_shared_goals:
            for group in self.config.agents:
                self.goal_pool.append(group.goal_pos)

        self.possible_agents = list(self.agents_dict.keys())
        self.agents = self.possible_agents.copy()

        self._setup_spaces()

        self.render_mode = render_mode
        self.window = None
        if render_mode == "human" or render_mode == "rgb_array":
            headless = render_mode == "rgb_array"
            self.window = SimulationWindow(
                target_fps=30, record=True, headless=headless
            )
            
        self.stuck_check_interval = 10
        self.stuck_min_dist = 0.02
        self.agent_stuck_points = {agent_id: [] for agent_id in self.agents}
        self.agent_last_positions = {agent_id: agent.pos.copy() for agent_id, agent in self.agents_dict.items()}
        self.agent_prev_goal_dist = {agent_id: np.linalg.norm(agent.goal_pos - agent.pos) for agent_id, agent in self.agents_dict.items()}
        self.winning_group = None  # Tracks which group won the competitive race

    def preprocess_agent_configs(
        self, agent_configs: List[AgentConfig], num_agents_per_group: int
    ):
        processed_configs = []
        for group_idx, agent_config in enumerate(agent_configs):
            start_rect = agent_config.start_pos
            width = start_rect.width
            middle = start_rect.center.x
            spacing = width / num_agents_per_group

            for i in range(num_agents_per_group):
                new_agent_config = agent_config.model_copy()
                if i == 0:
                    x_offset = 0
                elif i % 2 == 1:
                    x_offset = -((i + 1) // 2) * spacing
                else:
                    x_offset = (i // 2) * spacing

                this_center = Vector2(x=middle + x_offset, y=start_rect.center.y)
                new_rect = Rectangle(
                    center=this_center,
                    width=(width / num_agents_per_group - new_agent_config.radius * 2),
                    height=start_rect.height,
                )
                new_agent_config.start_pos = new_rect
                processed_configs.append(new_agent_config)
        return processed_configs

    def _setup_spaces(self):
        max_speed = max(agent.config.max_speed for agent in self.agents_dict.values())
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        state_dim = self.agent_states_dim
        lidar_dim = self.config.num_rays * 3

        obs_low = np.concatenate([np.array([-1] * state_dim), np.zeros(lidar_dim)])
        obs_high = np.concatenate([np.array([1] * state_dim), np.full(lidar_dim, max_speed)])
        self._observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    @property
    def agent_states_dim(self):
        return self.state_dim + 3

    @property
    def lidar_dim(self):
        return self.config.num_rays * 3

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def is_point_colliding(self, pos: np.ndarray, starting_points: List[np.ndarray], radius: float) -> bool:
        for starting_point in starting_points:
            if np.linalg.norm(pos - starting_point) < radius:
                return True
        return False

    def _point_in_rectangle(self, pos: np.ndarray, rect: Rectangle, margin: float = 0.0) -> bool:
        half_w = rect.width / 2 + margin
        half_h = rect.height / 2 + margin
        return (abs(pos[0] - rect.center.x) <= half_w and abs(pos[1] - rect.center.y) <= half_h)

    def _agent_in_circle(self, agent, circle) -> bool:
        dist = np.linalg.norm(agent.pos - circle.center.to_numpy())
        return dist <= (circle.radius + agent.radius * 0.5)

    def _agent_in_rectangle(self, agent, rect: Rectangle) -> bool:
        return self._point_in_rectangle(agent.pos, rect, margin=agent.radius * 0.5)

    def _agent_in_zone(self, agent, zone) -> bool:
        if hasattr(zone, 'type') and zone.type == "circle":
            return self._agent_in_circle(agent, zone)
        return self._agent_in_rectangle(agent, zone)

    def _update_switch_gate_state(self):
        """No-op: gates and triggers removed. Kept for compatibility."""
        self.active_switches = set()
        self.gate_open_mask = []
        self.gate_just_opened = False

    def _active_obstacles(self):
        obstacles = list(self.obstacles)
        for gate_open, gate_obstacle in zip(self.gate_open_mask, self.gate_obstacles):
            if not gate_open:
                obstacles.append(gate_obstacle)
        return obstacles

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        starting_points = []
        for agent in self.agents_dict.values():
            pos = sample_point_in_rectangle(agent.config.start_pos)
            num_tries = 0
            while (self.is_point_colliding(pos, starting_points, agent.radius*2.25) and num_tries < 10):
                pos = sample_point_in_rectangle(agent.config.start_pos)
                num_tries += 1
            starting_points.append(pos)
            agent.pos = pos
            agent.goal_pos = sample_point_in_rectangle(agent.goal_sample_area)
            agent.current_speed = 0.1
            _, agent.direction = convert_to_polar(agent.goal_pos - agent.pos)
            agent.active = True
            agent.goal_reached = False
            agent.lidar_observation_history.clear()
            agent.last_raw_lidar_observation = None
            agent.old_pos = agent.pos.copy()

        for obs in self.obstacles:
            if hasattr(obs, "reset"):
                obs.reset()

        self.num_steps = 0
        self.agents = self.possible_agents.copy()
        self.num_dead_agents = 0
        self.active_switches = set()
        self.gate_open_mask = [False for _ in self.gates]
        self.gate_just_opened = False
        self.group_goal_steps = {agent_id: None for agent_id in self.agents_dict}
        self.group_bonus_awarded = {group_idx: False for group_idx in self.group_members}
        self.pending_group_bonus = {agent_id: 0.0 for agent_id in self.agents_dict}
        self.agent_stuck_points = {agent_id: [] for agent_id in self.agents}
        self.agent_last_positions = {agent_id: agent.pos.copy() for agent_id, agent in self.agents_dict.items()}
        self.agent_prev_goal_dist = {agent_id: np.linalg.norm(agent.goal_pos - agent.pos) for agent_id, agent in self.agents_dict.items()}
        self.winning_group = None  # Reset competitive result each episode
        
        self.occupied_goals = set()
        if self.config.use_shared_goals:
            self.current_goal_locations = [sample_point_in_rectangle(goal_rect) for goal_rect in self.goal_pool]

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def _get_observations(self):
        self._update_switch_gate_state()
        lidar_observations = self.get_lidar_observation()
        for agent_id, lidar_observation in zip(self.agents, lidar_observations):
            self.agents_dict[agent_id].last_raw_lidar_observation = lidar_observation

        processed_lidar_observations = [
            self.process_lidar_observation(self.agents_dict[agent_id].config.max_range, lidar_observation)
            for agent_id, lidar_observation in zip(self.agents, lidar_observations)
        ]

        observations = {}
        for agent_idx, agent_id in enumerate(self.agents):
            agent = self.agents_dict[agent_id]
            state_dict = agent.get_state_dict()

            # Cooperative/competitive awareness features (must be 3 to match encoder):
            #  1. Fraction of all agents that have arrived
            #  2. Own distance to goal
            #  3. Whether winning group is decided (competitive signal)
            num_arrived = sum(1 for a in self.agents_dict.values() if a.goal_reached) / max(1, self.n_agents)
            curr_dist = np.linalg.norm(agent.goal_pos - agent.pos)
            race_decided = float(self.winning_group is not None) if hasattr(self, 'winning_group') else 0.0
            cooperative_features = np.array([num_arrived, curr_dist, race_decided], dtype=np.float32)

            state_vector = np.concatenate([np.array(state_dict["state_vector"], dtype=np.float32), cooperative_features])
            lidar_vector = processed_lidar_observations[agent_idx].flatten().astype(np.float32)
            observations[agent_id] = np.concatenate([state_vector, lidar_vector])

        return observations

    def calculate_reward(self, agent: Agent, collision_data: CollisionData):
        if agent.goal_reached:
            return 100.0
        if collision_data.is_colliding:
            # Soft penalty for bumping teammates (they need to move together)
            # Hard penalty for hitting walls/obstacles (must avoid these)
            if collision_data.colliding_with == "agent":
                return -2.0
            return -10.0

        # 1. Potential-based shaping: dense gradient toward goal
        curr_dist = np.linalg.norm(agent.goal_pos - agent.pos)
        prev_dist = self.agent_prev_goal_dist.get(agent.agent_id, curr_dist)
        potential_reward = (prev_dist - curr_dist) * self.config.potential_shaping_scale
        self.agent_prev_goal_dist[agent.agent_id] = curr_dist

        # 2. Small time penalty to encourage speed
        reward = potential_reward - 0.02

        # 3. Stuck zone memory penalty: discourage revisiting stuck spots
        for stuck_pos in self.agent_stuck_points.get(agent.agent_id, []):
            dist = np.linalg.norm(agent.pos - stuck_pos)
            if dist < 0.05:
                reward -= (0.05 - dist) * self.config.stuck_zone_penalty

        # 4. Simultaneous arrival group bonus (applied from pending after _update_group_arrival_bonus)
        reward += self.pending_group_bonus.get(agent.agent_id, 0.0)

        # 5. Formation cohesion: keep all agents moving as a group
        if self.config.formation_reward_weight > 0:
            group_idx = self.agent_group_map[agent.agent_id]
            members = self.group_members[group_idx]
            if len(members) > 1:
                centroid = np.mean([self.agents_dict[aid].pos for aid in members], axis=0)
                dist_from_centroid = np.linalg.norm(agent.pos - centroid)
                reward -= self.config.formation_reward_weight * max(0.0, dist_from_centroid - 0.08)

        return reward

    def _update_group_arrival_bonus(self):
        if self.config.simultaneous_arrival_bonus <= 0 or self.config.simultaneous_arrival_window <= 0:
            return
        for agent_id, agent in self.agents_dict.items():
            if agent.goal_reached and self.group_goal_steps[agent_id] is None:
                self.group_goal_steps[agent_id] = self.num_steps
        for group_idx, members in self.group_members.items():
            if self.group_bonus_awarded[group_idx]: continue
            goal_steps = [self.group_goal_steps[aid] for aid in members]
            if all(s is not None for s in goal_steps):
                if max(goal_steps) - min(goal_steps) <= self.config.simultaneous_arrival_window:
                    self.group_bonus_awarded[group_idx] = True
                    for aid in members: self.pending_group_bonus[aid] += self.config.simultaneous_arrival_bonus

    def _update_competitive_result(self):
        """When first agent reaches goal: bonus for winning team, penalty for losers."""
        if self.winning_group is not None:
            return  # Already resolved
        if not (self.config.winner_team_bonus != 0 or self.config.loser_team_penalty != 0):
            return
        for agent_id, agent in self.agents_dict.items():
            if agent.goal_reached:
                winning_group = self.agent_group_map[agent_id]
                self.winning_group = winning_group
                # Reward winning team's other members
                for aid in self.group_members[winning_group]:
                    if aid != agent_id:
                        self.pending_group_bonus[aid] += self.config.winner_team_bonus
                # Penalize all losing team members
                for group_idx, members in self.group_members.items():
                    if group_idx != winning_group:
                        for aid in members:
                            self.pending_group_bonus[aid] += self.config.loser_team_penalty
                break

    def step(self, actions: dict[str, np.ndarray]):
        if isinstance(actions, dict):
            processed_actions = [Vector2(x=float(actions.get(a, [0,0])[0]), y=float(actions.get(a, [0,0])[1])) for a in self.agents]
        else:
            processed_actions = [Vector2(x=float(a[0]), y=float(a[1])) for a in actions]
        self.num_steps += 1
        
        collision_datas = []
        terminations = {}
        truncations = {}
        
        for _ in range(self.config.repeat_steps):
            collision_datas, terminations, truncations = self.transition(processed_actions)
            
            # Strategy: any (Episode ends if one finishes)
            if self.config.terminal_strategy == "any":
                if any(terminations.values()) or any(truncations.values()):
                    break
            # Strategy: all/individual (Episode ends ONLY if ALL finish)
            else:
                if all(terminations.values()) or all(truncations.values()):
                    break
                    
        self._update_group_arrival_bonus()
        self._update_competitive_result()
        rewards = {aid: self.calculate_reward(self.agents_dict[aid], cd) for aid, cd in zip(self.agents, collision_datas)}
        for aid, rew in rewards.items():
            self.agents_dict[aid].last_reward = rew
            self.pending_group_bonus[aid] = 0.0
        if self.config.use_shared_goals:
            for aid, cd in zip(self.agents, collision_datas):
                ag = self.agents_dict[aid]
                if ag.goal_reached:
                    dists = [np.linalg.norm(loc - ag.pos) for loc in self.current_goal_locations]
                    idx = np.argmin(dists)
                    if idx not in self.occupied_goals: self.occupied_goals.add(idx)
        observations = self._get_observations()
        infos = {aid: {} for aid in self.agents}
        if self.render_mode == "human": self.render()
        return observations, rewards, terminations, truncations, infos

    def transition(self, actions: list[Vector2]):
        terminations, truncations = {}, {}
        for agent_id, action in zip(self.agents, actions):
            self.agents_dict[agent_id].apply_target_velocity(Vector2(x=action.x * self.agents_dict[agent_id].config.max_speed, y=action.y * self.agents_dict[agent_id].config.max_speed))
        for obs in self.obstacles: obs.update(DELTA_T)
        self._update_switch_gate_state()
        active_obstacles = self._active_obstacles()
        collision_datas = []
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            agent.update_pos(DELTA_T)
            if self.num_steps % self.stuck_check_interval == 0:
                dist_moved = np.linalg.norm(agent.pos - self.agent_last_positions[agent_id])
                if dist_moved < self.stuck_min_dist and not agent.goal_reached:
                    # Layer 2: Record stuck point for avoidance penalty
                    self.agent_stuck_points[agent_id].append(agent.pos.copy())
                    # Full random escape direction (360 degrees)
                    angle = np.random.uniform(0, 2 * np.pi)
                    agent.direction = np.array([np.cos(angle), np.sin(angle)])
                self.agent_last_positions[agent_id] = agent.pos.copy()
            cd = CollisionData()
            if self.boundary.violating_boundary(agent): cd.is_colliding, cd.colliding_with = True, "boundary"
            if not cd.is_colliding:
                for obs in active_obstacles:
                    if obs.check_collision(center=agent.pos, radius=agent.radius):
                        cd.is_colliding, cd.colliding_with = True, "obstacle"
                        break
            if not cd.is_colliding:
                for oid in self.agents:
                    if oid != agent_id and np.linalg.norm(self.agents_dict[oid].pos - agent.pos) < (agent.radius + self.agents_dict[oid].radius):
                        cd.is_colliding, cd.colliding_with = True, "agent"
                        break
            if cd.is_colliding:
                safe_pos = agent.old_pos.copy()
                delta = agent.pos - safe_pos
                agent.pos = safe_pos
                for test_delta in [np.array([delta[0], 0]), np.array([0, delta[1]])]:
                    if not self._is_colliding_with_anything(agent, safe_pos + test_delta, agent_id):
                        agent.pos = safe_pos + test_delta
                        if np.linalg.norm(test_delta) > 1e-6:
                            agent.direction = test_delta / np.linalg.norm(test_delta)
                        break
                else: # blocked
                    ang = np.random.uniform(-np.pi, np.pi)
                    agent.direction = np.array([np.cos(ang), np.sin(ang)])
            collision_datas.append(cd)
        for aid in self.agents:
            agent = self.agents_dict[aid]
            # Termination: Goal reached or Max Time
            # In 'all' or 'individual' strategy, we only mark individual agents as terminated.
            is_terminated = agent.goal_reached or self.num_steps >= self.config.max_time
            is_truncated = self.num_steps >= self.config.max_time
            
            terminations[aid] = np.bool_(is_terminated)
            truncations[aid] = np.bool_(is_truncated)
            
            # If agent has finished, it stays finished
            if is_terminated:
                agent.active = False
                agent.goal_reached = True

        return collision_datas, terminations, truncations

    def _is_colliding_with_anything(self, agent: Agent, pos: np.ndarray, agent_id: str) -> bool:
        orig = agent.pos
        agent.pos = pos
        coll = self.boundary.violating_boundary(agent)
        if not coll:
            for obs in self._active_obstacles():
                if obs.check_collision(center=pos, radius=agent.radius): coll = True; break
        if not coll:
            for oid in self.agents:
                if oid != agent_id and np.linalg.norm(self.agents_dict[oid].pos - pos) < (agent.radius + self.agents_dict[oid].radius): coll = True; break
        agent.pos = orig
        return coll

    def close(self):
        if self.window: self.window.close(); self.window = None

    def render(self):
        state = self.get_render_state()
        if self.render_mode in ["human", "rgb_array"]: return self.window.render(state)

    def process_lidar_observation(self, max_range, lidar_observation):
        """Vectorized processing of raw ray results into 3-channel LiDAR observation."""
        N = len(lidar_observation)
        if N == 0:
            return np.zeros((3, 0), dtype=np.float32)

        # Pre-allocate
        rays = np.zeros((3, N), dtype=np.float32)
        
        # Extract data efficiently
        intersects = np.array([r.intersects for r in lidar_observation])
        if not np.any(intersects):
            return rays

        ts = np.array([r.t if r.t is not None else 0.0 for r in lidar_observation])
        types = np.array([r.intersecting_with for r in lidar_observation])
        
        # Map types to channels (0: obstacle, 1: boundary, 2: agent)
        # Using vectorized boolean masks for speed
        rays[0, (types == "obstacle")] = max_range - ts[(types == "obstacle")]
        rays[1, (types == "boundary")] = max_range - ts[(types == "boundary")]
        rays[2, (types == "agent")] = max_range - ts[(types == "agent")]
        
        return rays

    def get_lidar_observation(self):
        """Generates LiDAR observations only for active agents to optimize performance."""
        all_rays, goals, rpa = [], [], []
        active_indices = []
        
        for i, aid in enumerate(self.agents):
            ag = self.agents_dict[aid]
            if ag.goal_reached:
                continue
                
            active_indices.append(i)
            all_rays.extend(create_lidar_rays(ag.pos, ag.direction, self.config.num_rays, ag.config.max_range, ag.config.fov_degrees))
            goals.append(Circle(center=Vector2(x=ag.goal_pos[0], y=ag.goal_pos[1]), radius=self.config.goal_threshold))
            rpa.append(self.config.num_rays)
        
        # Initialize full results with safe "NoHit" defaults
        final_results = [ [NoHit] * self.config.num_rays for _ in range(len(self.agents)) ]
        
        if all_rays:
            # Batch raycast for all ACTIVE agents at once
            res = batch_ray_intersection_detailed(
                np.array(all_rays), 
                self._active_obstacles(), 
                [self.config.boundary], 
                goals=goals, 
                agents=[Circle(center=Vector2(x=self.agents_dict[a].pos[0], y=self.agents_dict[a].pos[1]), radius=self.agents_dict[a].radius) for a in self.agents], 
                rays_per_agent=rpa
            )
            
            # Reshape and map back to the correct agent slots
            reshaped_res = np.reshape(res, (len(active_indices), self.config.num_rays))
            for local_idx, global_idx in enumerate(active_indices):
                final_results[global_idx] = reshaped_res[local_idx]
                
        return final_results

    def get_render_state(self):
        agents = [AgentState(position=(a.pos[0], a.pos[1]), radius=a.radius, color=a.config.agent_col, velocity=(a.current_speed*a.direction[0], a.current_speed*a.direction[1]), direction=(a.direction[0], a.direction[1]), lidar_observation=a.last_raw_lidar_observation, fov_degrees=a.config.fov_degrees, max_range=a.config.max_range, goals=Circle(center=Vector2(x=a.goal_pos[0], y=a.goal_pos[1]), radius=self.config.goal_threshold), goal_reached=a.goal_reached, last_reward=a.last_reward) for a in self.agents_dict.values()]
        switches = []
        for i, s in enumerate(self.switches):
            if isinstance(s.zone, Circle):
                switches.append(SwitchState(shape="circle", center=(s.zone.center.x, s.zone.center.y), radius=s.zone.radius, color=s.color, active=i in self.active_switches))
            else: # Rectangle
                switches.append(SwitchState(shape="rectangle", center=(s.zone.center.x, s.zone.center.y), width=s.zone.width, height=s.zone.height, rotation=s.zone.rotation, color=s.color, active=i in self.active_switches))
        
        return RenderState(agents=agents, obstacles=[o.get_current_state() for o in self._active_obstacles()], boundary=BoundaryState(vertices=[(v[0], v[1]) for v in self.boundary.vertices]), switches=switches)
