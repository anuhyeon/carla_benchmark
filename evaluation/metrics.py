import json
from pathlib import Path

import numpy as np


class Metrics:
    def __init__(self, start, goal, reference_length, target_count):
        self.last_position = np.asarray(start)[:2]
        self.goal = np.asarray(goal)[:2]
        self.reference_length = reference_length
        self.target_count = target_count
        self.path_length = 0.0
        self.collisions = 0

    def update(self, observation):
        position = np.asarray(observation["actor_position"])[:2]
        self.path_length += np.linalg.norm(position - self.last_position)
        self.last_position = position
        self.collisions += observation["collision_count"]

    def save(self, path, success, reached_targets, timeout):
        result = {
            "SR": int(success),
            "SPL": int(success)
            * self.reference_length
            / max(self.reference_length, self.path_length),
            "route_completion": reached_targets / self.target_count,
            "collision_count": self.collisions,
            "final_goal_error": float(np.linalg.norm(self.last_position - self.goal)),
            "timeout": timeout,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
