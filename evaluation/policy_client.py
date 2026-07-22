import atexit
import os
import subprocess
import sys
from pathlib import Path

import zmq


POLICY_DIR = Path(__file__).parents[1] / "policy"
sys.path.insert(0, str(POLICY_DIR))

from protocol import receive, send
from runtime_config import POLICY_RUNTIME

#### CARLA 평가 프로세스와 딥러닝 정책 프로세스를 분리하는 구조로 설계함 확장성 위해서 ####
class PolicyClient: # subprocess를 시작 (정책 전용 Python 환경으로 별도 worker 프로세스를 실행) #### # ZeroMQ IPC 소켓으로 그 프로세스에 연결하는 역할
    def __init__(self, name, config, checkpoint, mode): # 클라이언트 REQ 소켓 생성
        self.context = zmq.Context() #  컨텍스트 만들고
        self.socket = self.context.socket(zmq.REQ) # 제로엠큐 리퀘스트 요청 리퀘스트 소켓으로 worker 요청 보내고 응답 받음
        self.endpoint = f"ipc:///tmp/carla_policy_{os.getpid()}.sock" #  이 방식은 같은 컴퓨터에서만 통신할 수 있음 다른 컴퓨터의 worker와 통신하려면 tcp://호스트:포트 같은 endpoint가 필요
        self.process = subprocess.Popen( # worker 프로세스 실행 //// Popen은 worker가 종료될 때까지 기다리지 X
            [
                POLICY_RUNTIME[name]["python"],
                str(POLICY_DIR / "worker.py"),
                "--endpoint", self.endpoint,
                "--policy", name,
                "--config", str(config),
                "--checkpoint", str(checkpoint),
                "--mode", mode,
            ]
        )
        atexit.register(self._terminate_worker) # worker 프로세스의 종료 처리 등록 -> python 프로그램 종료 될때 자동으로 해당 함수 호출 약간 콜백함수 느낌
        self.socket.connect(self.endpoint) #  클라이언트가 endpoint 연결 등록 --> 클라이언트의 REQ 소켓이 IPC endpoint에 연결하도록 설정

    def _terminate_worker(self):
        self.process.terminate()
        self.process.wait()

    def request(self, message):
        send(self.socket, message) # protocol.py의 def send(socket, message):    socket.send_pyobj(message) 호출
        return receive(self.socket)

    def describe(self):
        return self.request({"type": "describe"})

    def reset(self, episode):
        return self.request({"type": "reset", "episode": episode})

    def step(self, observation, subgoal):
        return self.request(
            {"type": "step", "observation": observation, "subgoal": subgoal}
        )

    def close(self):
        self.request({"type": "close"})
        self.process.wait()
        self.socket.close()
        self.context.term()
# --policy citywalker이면 CityWalker 환경의 Python으로 별도 worker 프로세스 실행
