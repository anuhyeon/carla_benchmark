from collections import deque
from pathlib import Path
import sys

import numpy as np
import yaml


GENIE_ROOT = Path("/home/mobility/simulators/GENIE-SAMTP")
sys.path.insert(0, str(GENIE_ROOT))

from genie_path_planner.io_utils import resolve_path
from genie_path_planner.planner import PlannerConfig, plan_on_bev
from genie_path_planner.projection import (
    BEVObservation,
    blend_modalities,
    depth_to_bev_height_and_traversability,
    fuse_bev_observations,
    logits_to_traversability,
    project_score_to_bev,
)
from sam2.sam_tp import SAM_TP


class GeniePolicy:
    def __init__(self, config, checkpoint, mode="rgb", device="cuda"):
        config_path = Path(config).expanduser().resolve()
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.mode = mode

        samtp = self.config["samtp"]
        model_config = resolve_path(samtp["config_path"], config_path.parent, GENIE_ROOT)
        checkpoint_path = resolve_path(checkpoint, config_path.parent, GENIE_ROOT)
        self.model = SAM_TP(
            str(model_config),
            str(checkpoint_path),
            score_thresh=float(samtp.get("score_thresh", 0.0)),
            multimask=bool(samtp.get("multimask", False)),
        )
        self.model.sam2_model = self.model.sam2_model.to(device)
        self.score_transform = samtp.get("score_transform", "sigmoid")

        projection = self.config["projection"]
        self.resolution = float(projection["resolution_m_per_px"])
        self.forward_range = float(projection["forward_range_m"])
        self.side_range = float(projection["side_range_m"])
        self.max_ray_distance = float(projection["max_ray_distance_m"])

        self.depth_config = self.config["depth"]
        self.modality_fusion = self.config["fusion"]
        observation_fusion = self.config.get("observation_fusion", {})
        self.fusion_enabled = bool(observation_fusion.get("enabled", False))
        history_size = min(4, int(observation_fusion.get("max_observations", 4)))
        self.history = deque(maxlen=history_size if self.fusion_enabled else 1)
        self.planner_config = PlannerConfig(**self.config.get("planner", {}))

    def describe(self):
        sensors = ["rgb", "depth"] if self.mode == "rgbd" else ["rgb"]
        return {
            "sensors": sensors,
            "observation_interval": 1.0,
            "history_size": self.history.maxlen,
        }

    def reset(self, episode):
        self.history.clear()
        return {"status": "ok"}

    def step(self, observation, subgoal):
        camera_k = np.asarray(observation["camera_K"], dtype=np.float64)
        camera_pose = np.asarray(observation["T_world_camera"], dtype=np.float64)
        actor_pose = np.asarray(observation["T_world_actor"], dtype=np.float64)

        output = self.model.run_sam2_inference(
            np.asarray(observation["rgb"], dtype=np.uint8)
        )
        traversability = logits_to_traversability(
            output["logits"], transform=self.score_transform
        )
        rgb_bev, rgb_observed, _ = project_score_to_bev( # Depth 없이 카메라 높이를 이용해 BEV를 만드는 핵심 부분!!rgb_bev : 각 BEV cell의 통행 가능성 점수 //// rgb_observed : 각 BEV cell이 실제 RGB 관측으로 채워졌는지 나타내는 mask
            score_map=traversability,
            camera_k=camera_k,
            camera_pose=camera_pose,
            ground_z=float(observation["ground_z"]),
            bev_resolution_m_per_px=self.resolution,
            bev_forward_range_m=self.forward_range,
            bev_side_range_m=self.side_range,
            max_ray_distance_m=self.max_ray_distance,
        )

        depth_bev = None
        depth_observed = None
        rgbd_bev = None
        rgbd_observed = None
        if self.mode == "rgbd":
            _, depth_bev, depth_observed, _ = depth_to_bev_height_and_traversability(
                depth_m=np.asarray(observation["depth_m"], dtype=np.float32),
                camera_k=camera_k,
                camera_pose=camera_pose,
                ground_z=float(observation["ground_z"]),
                reliable_depth_m=float(self.depth_config["reliable_depth_m"]),
                min_depth_m=float(self.depth_config["min_depth_m"]),
                obstacle_height_thresh_m=float(
                    self.depth_config["obstacle_height_thresh_m"]
                ),
                bev_resolution_m_per_px=self.resolution,
                bev_forward_range_m=self.forward_range,
                bev_side_range_m=self.side_range,
            )
            rgbd_bev, rgbd_observed = blend_modalities(
                rgb_bev=rgb_bev,
                rgb_observed=rgb_observed,
                depth_bev=depth_bev,
                depth_observed=depth_observed,
                rgb_weight=float(self.modality_fusion["rgb_weight"]),
                depth_weight=float(self.modality_fusion["depth_weight"]),
                require_depth=bool(
                    self.modality_fusion.get("require_depth_for_rgbd", True)
                ),
            )

        record = BEVObservation( # 현재 한 번의 관측에서 계산된 RGB/depth 기반 BEV 결과와 해당 관측 시점의 pose를 BEVObservation 객체 하나로 묶는 코드
            name=str(observation["frame"]),
            camera_pose=camera_pose,
            robot_pose=actor_pose,
            bev_resolution_m=self.resolution,
            rgb_bev=rgb_bev,
            rgb_observed=rgb_observed,
            depth_bev=depth_bev,
            depth_observed=depth_observed,
            rgbd_bev=rgbd_bev,
            rgbd_observed=rgbd_observed,
        )
        self.history.append(record)
        records = list(self.history) if self.fusion_enabled else [record]
        bev, observed, _ = fuse_bev_observations( # 여기는 카메라 기준 BEV를 actor 기준으로 변경 그리고 fusion도 할수있도록 한거 같은데 과거 이미지를 기반으로 근데 지금은 안하는세팅
            records=records,
            mode=self.mode,
            reference_pose=actor_pose,
            reference_frame="base",
            bev_resolution_m=self.resolution,
            bev_forward_range_m=self.forward_range,
            bev_side_range_m=self.side_range,
        )

        goal_world = np.r_[np.asarray(subgoal, dtype=np.float64)[:3], 1.0]
        goal_base = np.linalg.inv(actor_pose) @ goal_world
        planned = plan_on_bev(
            bev_traversability=bev,
            observed_mask=observed,
            goal_x_m=-float(goal_base[1]),
            goal_y_m=float(goal_base[0]),
            bev_resolution_m=self.resolution,
            config=self.planner_config,
        )
        path = planned.final_path_xy_m # (101, 2)     [right, forward]
        if len(path) == 0:
            return {
                "ready": True,
                "status": "no_path",
                "path_m": np.empty((0, 2), dtype=np.float32),
            }
        return {
            "ready": True,
            "status": "ok",
            "path_m": path[:, [1, 0]].astype(np.float32), #[forward, right]
        }
