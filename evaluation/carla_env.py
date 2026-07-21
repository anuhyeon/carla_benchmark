import math
import queue

import carla
import numpy as np


FIXED_DELTA_SECONDS = 0.05
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 576
CAMERA_FOV = 120.0
CAMERA_GROUND_HEIGHT = 0.145

WORLD_FROM_CARLA = np.diag([1.0, -1.0, 1.0, 1.0])
SENSOR_FROM_OPTICAL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
ACTOR_FROM_BASE = np.diag([1.0, -1.0, 1.0, 1.0])


def camera_matrix(transform):
    return WORLD_FROM_CARLA @ np.asarray(transform.get_matrix()) @ SENSOR_FROM_OPTICAL


def actor_matrix(transform):
    return WORLD_FROM_CARLA @ np.asarray(transform.get_matrix()) @ ACTOR_FROM_BASE


def read_frame(data_queue, frame):
    data = data_queue.get()
    while data.frame < frame:
        data = data_queue.get()
    return data


def rgb_array(image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    )
    return bgra[:, :, :3][:, :, ::-1].copy()


def depth_array(image):
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        image.height, image.width, 4
    ).astype(np.float32)
    normalized = (
        bgra[:, :, 2] + 256.0 * bgra[:, :, 1] + 65536.0 * bgra[:, :, 0]
    ) / 16777215.0
    return normalized * 1000.0


class CarlaEnv:
    def __init__(self, episode, sensors):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(episode["map"])
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(settings)

        start = episode["start"]
        actor_bp = self.world.get_blueprint_library().find("vehicle.bh.crossbike")
        self.actor = self.world.spawn_actor(
            actor_bp,
            carla.Transform(
                carla.Location(x=start["x"], y=start["y"], z=start["z"] + 0.5),
                carla.Rotation(yaw=episode["start_yaw"]),
            ),
        )
        self.actor.apply_control(carla.VehicleControl(brake=1.0)) # 초기화 중 움직임 방지용 apply_control() --> 실제 제어 명령 적용
        for _ in range(round(1.0 / FIXED_DELTA_SECONDS)): # FIXED_DELTA_SECONDS = 0.05 
            self.world.tick() # 20 tick 동안 시뮬레이션을 진행 -> 시뮬 시간 1초

        camera_transform = carla.Transform(
            carla.Location(
                x=0.8,
                z=start["z"] + CAMERA_GROUND_HEIGHT - self.actor.get_location().z,
            )
        )
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        camera_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_FOV))
        self.rgb_camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.actor,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.rgb_queue = queue.Queue()
        self.rgb_camera.listen(self.rgb_queue.put) # CARLA RGB 카메라에 데이터 수신 callback을 등록 --. self.rgb_queue.put(image)랑 같은 의미

        self.depth_camera = None
        self.depth_queue = None
        if "depth" in sensors:
            depth_bp = self.world.get_blueprint_library().find("sensor.camera.depth")
            depth_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
            depth_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
            depth_bp.set_attribute("fov", str(CAMERA_FOV))
            self.depth_camera = self.world.spawn_actor(
                depth_bp,
                camera_transform,
                attach_to=self.actor,
                attachment_type=carla.AttachmentType.Rigid,
            )
            self.depth_queue = queue.Queue()
            self.depth_camera.listen(self.depth_queue.put)

        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.actor
        )
        self.collision_events = queue.Queue()
        self.last_collision_frame = {}
        self.collision_sensor.listen(self.collision_events.put)

        focal = CAMERA_WIDTH / (2.0 * math.tan(math.radians(CAMERA_FOV) / 2.0))
        self.camera_k = np.array(
            [
                [focal, 0.0, CAMERA_WIDTH / 2.0],
                [0.0, focal, CAMERA_HEIGHT / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def collision_count(self):
        count = 0
        while not self.collision_events.empty():
            event = self.collision_events.get()
            actor_id = event.other_actor.id
            last_frame = self.last_collision_frame.get(actor_id)
            if last_frame is None or event.frame > last_frame + 1:
                count += 1
            self.last_collision_frame[actor_id] = event.frame
        return count

    def tick(self):
        frame = self.world.tick() # CARLA 월드를 한 프레임 진행
        rgb = read_frame(self.rgb_queue, frame) # 현재 frame의 RGB 이미지 수신
        depth = read_frame(self.depth_queue, frame) if self.depth_queue else None
        snapshot = self.world.get_snapshot() # snapshot은 특정 시점의 월드 상태를 고정해서 보여주는 객체 -> frame 번호,시뮬레이션 timestamp,월드에 존재하는 actor들의 상태,actor 위치와 회전,actor 속도와 가속도
        actor_snapshot = snapshot.find(self.actor.id)
        camera_snapshot = snapshot.find(self.rgb_camera.id)
        actor_pose = actor_matrix(actor_snapshot.get_transform()) # actor_snapshot.get_transform() 로 현재 frame에서 차량의 위치와 회전을 가져오고(actor 로컬 좌표계의 원점이 현재 CARLA world 좌표계에서 어디에 있는가?) actor_matrix()는 이를 프로젝트에서 사용하는 4×4 변환 행렬로 바쑴 --> actor의 로컬 좌표 점을 프로젝트 내부 world 좌표로 변환하는 행렬
        camera_pose = camera_matrix(camera_snapshot.get_transform())
        velocity = actor_snapshot.get_velocity()

        observation = { # 한 번의 CARLA tick에서 얻은 센서 데이터와 차량 상태를 observation 딕셔너리로 묶는부분
            "frame": frame,
            "simulation_time": snapshot.timestamp.elapsed_seconds,
            "rgb": rgb_array(rgb),
            "camera_K": self.camera_k,
            "T_world_camera": camera_pose,
            "T_world_actor": actor_pose,
            "actor_position": actor_pose[:3, 3].copy(),
            "speed": math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            "collision_count": self.collision_count(),
            "ground_z": float(camera_pose[2, 3] - CAMERA_GROUND_HEIGHT),
        }
        if depth is not None:
            observation["depth_m"] = depth_array(depth)
        return observation

    def apply_control(self, control):
        self.actor.apply_control(control)

    def close(self):
        sensors = [self.rgb_camera, self.depth_camera, self.collision_sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.actor.destroy()
        self.world.apply_settings(self.original_settings)
