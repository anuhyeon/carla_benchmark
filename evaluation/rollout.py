#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from carla_env import FIXED_DELTA_SECONDS, CarlaEnv
from controller import Controller
from metrics import Metrics
from policy_client import PolicyClient


WARMUP_SECONDS = 5.0


def world_location(location):
    return np.array([location["x"], -location["y"], location["z"]])


def load_episode(path):
    path = Path(path)
    metadata = json.loads((path / "metadata.json").read_text())
    first_pose_file = next(iter(sorted((path / "pose").glob("*.json"))))
    first_pose = json.loads(first_pose_file.read_text())
    subgoals = [
        world_location(json.loads(file.read_text())["location"])
        for file in sorted((path / "subgoal").glob("*.json"))
    ]
    goal = world_location(metadata["goal"])
    if np.linalg.norm(subgoals[-1][:2] - goal[:2]) > 0.01:
        subgoals.append(goal)
    start = world_location(metadata["start"])
    poses = [
    world_location(json.loads(file.read_text())["walker"])
        for file in sorted(
            (path / "pose").glob("*.json"),
            key=lambda file: int(file.stem.split("_")[1]),
        )
    ]
    # route = [start, *subgoals]
    route = [start, *poses, goal]
    reference_length = sum(
        np.linalg.norm(current[:2] - previous[:2])
        for previous, current in zip(route, route[1:])
    )
    return {
        **metadata,
        "path": path,
        "start_yaw": first_pose["walker"]["yaw"],
        "start_world": start,
        "goal_world": goal,
        "subgoals_world": subgoals,
        "reference_length": reference_length,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode")
    parser.add_argument("--policy", choices=["citywalker", "genie_samtp"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument("--subgoal-radius", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--waypoint-index", type=int, default=0)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    if args.display:
        from visualizer import close, draw, setup

    episode = load_episode(args.episode)
    policy = PolicyClient(args.policy, args.config, args.checkpoint, args.mode) # PolicyClient 생성 
    requirements = policy.describe() # worker에게 요청 보냄 
    policy.reset({"step_scale": episode["step_scale"]})
    env = CarlaEnv(episode, requirements["sensors"])
    controller = Controller(args.waypoint_index)
    targets = episode["subgoals_world"]
    inference_ticks = round(
        requirements["observation_interval"] / FIXED_DELTA_SECONDS
    )
    ########### 본격적인 평가를 시작하기 전에 CARLA를 5초 동안 진행하면서 정책의 observation history와 모델 실행 상태를 준비하는 warmup 구간 ###############
    warmup_ticks = round(WARMUP_SECONDS / FIXED_DELTA_SECONDS)
    for tick in range(warmup_ticks): # 5초
        observation = env.tick()
        if tick % inference_ticks == 0:
            prediction = policy.step(observation, targets[0])
        control = controller.compute(observation["T_world_actor"], observation["speed"]) # compute()는 path가 비어 있으면 최대 브레이크를 반환 현재 코드에서는 warmup 동안 경로를 Controller에 넣지 않기 때문에 차량은 브레이크를 건 상태로 정지
        env.apply_control(control)
    #######################################################################
    metrics = Metrics(
        observation["actor_position"],
        episode["goal_world"],
        episode["reference_length"],
        len(targets),
    )
    target_index = 0
    success = False
    user_quit = False
    max_ticks = round(args.timeout / FIXED_DELTA_SECONDS)
    planner_width = (
        prediction["planner_visualization"].shape[1]
        if args.policy == "genie_samtp"
        else 0
    )
    display = (
        setup(observation["rgb"].shape[1], observation["rgb"].shape[0], planner_width)
        if args.display
        else None
    )

    for tick in range(max_ticks):
        observation = env.tick()
        metrics.update(observation)

        if np.linalg.norm(
            observation["actor_position"][:2] - targets[target_index][:2]
        ) <= args.subgoal_radius:
            target_index += 1
            controller.clear()
            if target_index == len(targets):
                success = True
                break

        if tick % inference_ticks == 0:
            prediction = policy.step(observation, targets[target_index])
            if prediction["ready"] and prediction["status"] == "ok":
                controller.set_path(prediction["path_m"], observation["T_world_actor"])
            elif prediction["ready"]:
                controller.clear()

        control = controller.compute(observation["T_world_actor"], observation["speed"])
        env.apply_control(control)

        if display and not draw(
            *display,
            observation,
            targets,
            target_index,
            controller.path_world,
            args.policy,
            prediction,
            (tick + 1) * FIXED_DELTA_SECONDS,
            args.timeout,
            metrics.collisions,
        ):
            user_quit = True
            break

    output = (
        Path(__file__).parent
        / "results"
        / episode["name"]
        / args.policy
        / "metrics.json"
    )
    result = metrics.save(output, success, target_index, not success and not user_quit)
    print(json.dumps(result, indent=2))
    if display:
        close()
    policy.close()
    env.close()


if __name__ == "__main__":
    main()
