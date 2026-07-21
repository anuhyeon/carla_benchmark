import math

import carla
import numpy as np


class Controller:
    def __init__(self, waypoint_index, target_speed=1.4):
        self.waypoint_index = waypoint_index # 내가 지정한 N 번째 waypoint 하ㅏ나
        self.target_speed = target_speed # 목표 속도는  1.4m/s 
        self.path_world = np.empty((0, 3)) # policy가 예측한 waypoint를 월드 좌표로 저장 하기 위한 세팅

    def set_path(self, path_m, actor_pose): # actor_pose --> 추론 시점의 T_world_actor --> Actor 좌표를 월드 좌표로 변환하는 4×4 행렬
        path_m = np.asarray(path_m) # policy가 예측한 N×2 waypoint 배열
        points = np.column_stack(
            [path_m[:, 0], -path_m[:, 1], np.zeros(len(path_m)), np.ones(len(path_m))] #  path_m[:, 0] = Actor 전방 거리     path_m[:, 1] = Actor 오른쪽 거리
        )
        self.path_world = (actor_pose @ points.T).T[:, :3] ########### 추론 시점의 Actor 기준 waypoint를 월드 좌표로 변환 #############3
    def clear(self):
        self.path_world = np.empty((0, 3))

    def compute(self, actor_pose, speed):
        if len(self.path_world) == 0: # self.path_world = np.empty((0, 3)) 이면 
            return carla.VehicleControl(brake=1.0)
        ############ 월드 waypoint를 현재 Actor 좌표로 재변환 ###############
        points = np.column_stack(
            [self.path_world, np.ones(len(self.path_world))]
        )
        local = (np.linalg.inv(actor_pose) @ points.T).T
        path = np.column_stack([local[:, 0], -local[:, 1]]) # [전방, 오른쪽]
        ##################################################################
        target = path[self.waypoint_index] # 지정한 waypoiny 선택
        angle = math.atan2(target[1], target[0]) # 목표 방향으로 조향
        steer = float(np.clip(angle / math.radians(45.0), -1.0, 1.0)) # CARLA의 steer 삼지창 [-1, 1] 범위로 변환 -->목표 방향각을 단순히 정규화한 비례 제어값
        speed_error = self.target_speed - speed #목표 속도와 현재 속도의 차이를 계산
        throttle = float(np.clip(speed_error, 0.0, 1.0))
        brake = float(np.clip(-speed_error, 0.0, 1.0))
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

# actor_pose는 현재 차량이 월드의 어디에 있고 어느 방향을 보고 있는지를 하나의 행렬로 표현한 값     --e> actor_pose = 월드 원점 기준 차량의 위치 + 방향
# T_world_actor = Actor 좌표의 점을 World 좌표로 변환하는 행렬