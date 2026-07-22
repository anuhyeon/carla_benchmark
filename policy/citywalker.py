import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from runtime_config import POLICY_RUNTIME

CITYWALKER_ROOT = Path(POLICY_RUNTIME["citywalker"]["project_root"])
sys.path.insert(0, str(CITYWALKER_ROOT))

from model.citywalker_feat import CityWalkerFeat


class DictNamespace(argparse.Namespace):
    def __init__(self, **values):
        super().__init__(
            **{
                key: DictNamespace(**value) if isinstance(value, dict) else value
                for key, value in values.items()
            }
        )


DictNamespace.__module__ = "__main__"


class CityWalkerPolicy:
    def __init__(self, config, checkpoint, device="cuda"):
        cfg = DictNamespace(**yaml.safe_load(Path(config).read_text()))
        self.device = torch.device(device)
        self.model = CityWalkerFeat(cfg)

        with torch.serialization.safe_globals([DictNamespace]):
            checkpoint_data = torch.load(
                checkpoint, map_location="cpu", weights_only=True
            )
        state = {
            name.removeprefix("model."): value
            for name, value in checkpoint_data["state_dict"].items()
        }
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.history = deque(maxlen=5)
        self.step_scale = 1.0

    def describe(self):
        return {
            "sensors": ["rgb"],
            "observation_interval": 1.0,
            "history_size": 5,
        }

    def reset(self, episode):
        self.history.clear()
        self.step_scale = float(episode["step_scale"])
        return {"status": "ok"}

    def step(self, observation, subgoal):
        rgb = torch.from_numpy(observation["rgb"]).permute(2, 0, 1).float() / 255.0
        rgb = F.interpolate(
            rgb.unsqueeze(0), size=(360, 640), mode="bilinear", align_corners=False
        ).squeeze(0)
        camera_pose = np.asarray(observation["T_world_camera"], dtype=np.float64)
        self.history.append((rgb, camera_pose))

        if len(self.history) < 5:
            return {"ready": False, "status": "warming_up"}

        images = torch.stack([item[0] for item in self.history]).unsqueeze(0)
        images = images.to(self.device)

        latest_camera = self.history[-1][1]
        camera_from_world = np.linalg.inv(latest_camera)
        history_positions = np.stack(
            [(camera_from_world @ item[1])[:3, 3][[0, 2]] for item in self.history]
        )

        goal_world = np.append(np.asarray(subgoal, dtype=np.float64), 1.0)
        goal_position = (camera_from_world @ goal_world)[[0, 2]]
        coordinates = np.concatenate([history_positions, goal_position[None]])
        coordinates = torch.from_numpy(coordinates / self.step_scale).float()
        coordinates = coordinates.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            waypoints, arrival_logit, _, _ = self.model(images, coordinates)

        waypoints = waypoints[0].cpu().numpy() * self.step_scale
        path_m = waypoints[:, [1, 0]].astype(np.float32)

        return {
            "ready": True,
            "status": "ok",
            "path_m": path_m,
            "arrival_score": torch.sigmoid(arrival_logit).item(),
        }
    
        # waypoints = waypoints[0].cpu().numpy() * self.step_scale
            # camera_points = np.column_stack(
            #     [waypoints[:, 0], np.zeros(len(waypoints)), waypoints[:, 1], np.ones(len(waypoints))]
            # )
            # camera_points = np.column_stack(
            #     [waypoints[:, 0], np.zeros(len(waypoints)), waypoints[:, 1], np.zeros(len(waypoints))]
            # )
            # actor_from_world = np.linalg.inv(
            #     np.asarray(observation["T_world_actor"], dtype=np.float64)
            # )
            # actor_points = (actor_from_world @ latest_camera @ camera_points.T).T

            # return {
            #     "ready": True,
            #     "status": "ok",
            #     "path_m": np.column_stack(
            #         [actor_points[:, 0], -actor_points[:, 1]]
            #     ).astype(np.float32),
            #     "arrival_score": torch.sigmoid(arrival_logit).item(),
            # }
