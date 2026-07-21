#!/usr/bin/env python3

import argparse
import json
import math
import queue
import random
from pathlib import Path

import carla


FIXED_DELTA_SECONDS = 0.05
IMAGE_INTERVAL_SECONDS = 1.0
SUBGOAL_INTERVAL_METERS = 5.0
MIN_GOAL_DISTANCE_METERS = 10.0
GOAL_THRESHOLD_METERS = 1.0
MAX_EPISODE_SECONDS = 300.0
WALKER_SPEED = 1.4


def location_data(location):
    return {"x": location.x, "y": location.y, "z": location.z}


def transform_data(transform):
    return {
        **location_data(transform.location),
        "roll": transform.rotation.roll,
        "pitch": transform.rotation.pitch,
        "yaw": transform.rotation.yaw,
    }


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(__file__).parent / args.name
    image_dir = output_dir / "image"
    pose_dir = output_dir / "pose"
    subgoal_dir = output_dir / "subgoal"
    image_dir.mkdir(parents=True)
    pose_dir.mkdir()
    subgoal_dir.mkdir()

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    original_settings = world.get_settings()
    actors = []
    controller = None
    camera = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        world.set_pedestrians_seed(args.seed)
        world.set_pedestrians_cross_factor(0.0)

        start = world.get_random_location_from_navigation()
        goal = world.get_random_location_from_navigation()
        while start.distance(goal) < MIN_GOAL_DISTANCE_METERS:
            goal = world.get_random_location_from_navigation()

        yaw = math.degrees(math.atan2(goal.y - start.y, goal.x - start.x))
        walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        walker = world.spawn_actor(
            walker_bp,
            carla.Transform(start, carla.Rotation(yaw=yaw)),
        )
        actors.append(walker)

        controller_bp = world.get_blueprint_library().find("controller.ai.walker")
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        actors.append(controller)
        world.tick()

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb") # 카메라 부착 위치
        camera_bp.set_attribute("image_size_x", "1024")
        camera_bp.set_attribute("image_size_y", "576")
        camera_bp.set_attribute("fov", "120")
        camera_z = start.z + 0.145 - walker.get_location().z
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=0.5, z=camera_z)),
            attach_to=walker,
            attachment_type=carla.AttachmentType.Rigid,
        )
        actors.append(camera)
        images = queue.Queue()
        camera.listen(images.put)

        controller.start()
        controller.go_to_location(goal)
        controller.set_max_speed(WALKER_SPEED)

        image_interval_ticks = round(IMAGE_INTERVAL_SECONDS / FIXED_DELTA_SECONDS)
        max_ticks = round(MAX_EPISODE_SECONDS / FIXED_DELTA_SECONDS)
        previous_location = walker.get_location()
        route_distance = 0.0
        next_subgoal_distance = SUBGOAL_INTERVAL_METERS
        subgoal_index = 0
        previous_camera_position = None
        step_distance = 0.0
        step_count = 0

        for tick in range(max_ticks):
            frame = world.tick()
            image = images.get()
            while image.frame < frame:
                image = images.get()

            snapshot = world.get_snapshot()
            walker_snapshot = snapshot.find(walker.id)
            camera_snapshot = snapshot.find(camera.id)
            walker_transform = walker_snapshot.get_transform()
            camera_transform = camera_snapshot.get_transform()
            current_location = walker_transform.location
            route_distance += previous_location.distance(current_location)
            previous_location = current_location

            if tick % image_interval_ticks == 0:
                camera_position = (camera_transform.location.x, camera_transform.location.y)
                if previous_camera_position is not None:
                    step_distance += math.hypot(
                        camera_position[0] - previous_camera_position[0],
                        camera_position[1] - previous_camera_position[1],
                    )
                    step_count += 1
                previous_camera_position = camera_position
                filename = f"frame_{frame:06d}"
                image.save_to_disk(str(image_dir / f"{filename}.png"))
                save_json(
                    pose_dir / f"{filename}.json",
                    {
                        "frame": frame,
                        "simulation_time": tick * FIXED_DELTA_SECONDS,
                        "camera": transform_data(camera_transform),
                        "walker": transform_data(walker_transform),
                    },
                )

            if route_distance >= next_subgoal_distance:
                save_json(
                    subgoal_dir / f"subgoal_{subgoal_index:04d}.json",
                    {
                        "index": subgoal_index,
                        "frame": frame,
                        "distance_from_start": route_distance,
                        "location": location_data(current_location),
                    },
                )
                subgoal_index += 1
                next_subgoal_distance += SUBGOAL_INTERVAL_METERS

            goal_distance = math.hypot(
                current_location.x - goal.x,
                current_location.y - goal.y,
            )
            if goal_distance < GOAL_THRESHOLD_METERS:
                break

        if goal_distance >= GOAL_THRESHOLD_METERS:
            goal = current_location
        save_json(
            subgoal_dir / f"subgoal_{subgoal_index:04d}.json",
            {
                "index": subgoal_index,
                "frame": frame,
                "distance_from_start": route_distance,
                "location": location_data(goal),
            },
        )

        save_json(
            output_dir / "metadata.json",
            {
                "name": args.name,
                "map": world.get_map().name.split("/")[-1],
                "seed": args.seed,
                "start": location_data(start),
                "goal": location_data(goal),
                "image_interval_seconds": IMAGE_INTERVAL_SECONDS,
                "subgoal_interval_meters": SUBGOAL_INTERVAL_METERS,
                "step_scale": step_distance / step_count,
                "route_distance": route_distance,
                "duration": tick * FIXED_DELTA_SECONDS,
            },
        )
    finally:
        if camera is not None:
            camera.stop()
        if controller is not None:
            controller.stop()
        for actor in reversed(actors):
            actor.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
