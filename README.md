
## 1. CARLA 설치

[CARLA Quick Start](https://carla.readthedocs.io/en/latest/start_quickstart/)를 따라 CARLA server와 Python client library 설치를 먼저 완료하면 됨.

CARLA 설치 후 아래 구조가 있어야 함.

```text
CARLA_0.9.16/
├── CarlaUE4.sh
└── PythonAPI/
    └── carla/
        └── dist/
```

이걸로 잘 서버 켜지는지 확인
```
./CarlaUE4.sh
```
그리고 추가적으로 세팅한 CARLA 가상 환경에 아래도 설치 
```
python -m pip install pyzmq
``` 

## 2. Benchmark clone

CARLA의 `PythonAPI` 디렉터리 아래에 아래 저장소를 clone.

```bash
cd /absolute/path/to/CARLA_0.9.16/PythonAPI
git clone https://github.com/anuhyeon/carla_benchmark.git benchmark
```
에피소드 zip파일은 아래서 풀면 됨.
`PythonAPI/benchmark/episodes/` 

## 3. Policy 세팅: GENIE-SAMTP 예시

평가할 policy 프로젝트의 위치는 자유. 

[jiaming-ai/GENIE-SAMTP](https://github.com/jiaming-ai/GENIE-SAMTP)

위 깃허브의 GENIE 세팅 완료후

마찬가지로 worker 통신에 필요한 pyzmq를 설치

```bash
python -m pip install pyzmq
```


## 4. Policy Python과 project root 설정

`PythonAPI/benchmark/policy/runtime_config.py`에 각 policy의 Python 실행 파일과 프로젝트 root 절대경로를 입력하면 됨.

```python
POLICY_RUNTIME = {
    "citywalker": {
        "python": "/absolute/path/to/conda/envs/citywalker/bin/python",
        "project_root": "/absolute/path/to/citywalker",
    },
    "genie_samtp": {
        "python": "/absolute/path/to/conda/envs/sam_tp/bin/python",
        "project_root": "/absolute/path/to/GENIE-SAMTP",
    },
}
```

- `python`은 policy worker를 실행할 가상환경의 Python 실행 파일
- `project_root`는 policy 디렉토러 위치


## 5. CARLA server 실행

### Headless server

화면이 없는 서버에서는 CARLA root에서 아래처럼 실행하면됨

```bash
cd /absolute/path/to/CARLA_0.9.16

./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound 
```

- `-RenderOffScreen`은 spectator 창을 띄우지 않지만 RGB와 depth 렌더링은 정상적으로 수행
- `-nosound`는 오디오 장치 초기화와 소리 출력을 비활성화



Hybrid Intel/NVIDIA 장비에서 NVIDIA Vulkan ICD를 명시해야 하는 경우 아래처럼 해야할 수 도 있음.

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
__NV_PRIME_RENDER_OFFLOAD=1 \
./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
```

## 6. GENIE-SAMTP rollout 실행

CARLA server가 완전히 실행된 뒤 다른 terminal에서 CARLA 평가 환경을 활성화하고 rollout을 실행하면 됨.

```bash
conda activate carla0916
cd /absolute/path/to/CARLA_0.9.16

python PythonAPI/benchmark/evaluation/rollout.py \
  PythonAPI/benchmark/episodes/town01_episode_001 \
  --policy genie_samtp \
  --config /absolute/path/to/GENIE-SAMTP/configs/stretch_path_planner.yaml \
  --checkpoint /absolute/path/to/GENIE-SAMTP/sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt \
  --mode rgb \
  --waypoint-index 25 \
  --subgoal-radius 2.0 \
  --subgoal-stride 1 \
  --timeout 600
```

위 명령은 Pygame 창을 띄우지 않는 headless rollout임. 로컬 화면에 RGB, subgoal map, GENIE planner visualization을 표시하려면 마지막에 `--display`를 추가하면 됨.

```bash
  --display
```

## 7. Rollout 인자

| 인자 | 필수 여부 | 기본값 | 역할 |
|---|---:|---:|---|
| `episode` | 필수 | 없음 | 평가할 episode 디렉터리 경로임. `--episode`가 아닌 첫 번째 위치 인자임. |
| `--policy` | 필수 | 없음 | `genie_samtp` 처럼  실행할 policy를 선택함. |
| `--config` | 필수 | 없음 | policy 모델 및 planner YAML 경로임. `policy/runtime_config.py`와는 다른 파일임. |
| `--checkpoint` | 필수 | 없음 | policy checkpoint 파일 경로임. GENIE-SAMTP config 내부 값보다 이 인자로 전달한 경로를 사용함. |
| `--mode` | 선택 | `rgb` | `rgb` 또는 `rgbd` 입력을 선택함. GENIE-SAMTP의 `rgbd`는 CARLA depth sensor도 생성함.|
| `--waypoint-index` | 선택 | `0` | policy가 예측한 local path에서 controller가 추종할 waypoint index임. 출력 경로 길이보다 작아야 함. |
| `--subgoal-radius` | 선택 | `2.0` | actor와 현재 subgoal 사이의 평면거리가 이 값 이하이면 도달한 것으로 판단하는 반경이며 단위는 m임. |
| `--subgoal-stride` | 선택 | `1` | 저장된 subgoal 중 매 N번째 target만 사용하는 값임. 1 이상이어야 하며 마지막 goal은 항상 포함함. |
| `--timeout` | 선택 | `300.0` | rollout 평가 제한 시뮬레이션 시간이며 단위는 초임. . |
| `--display` | 선택 | 비활성 | Pygame 평가 창을 활성화함. Headless server에서는 생략하면 됨. |

`--subgoal-stride`는 직선거리 간격의미X , episode에 저장된 subgoal 순서에서 N개마다 선택하는 값. 예를 들어 원본 subgoal이 누적 경로거리 약 5m마다 저장되었다면 stride 2는 누적 경로거리 약 10m마다 선택하는 방식

## 8. Episode 구조 

episode 디렉터리는 다음 구조 사용 (Image는 시각화용 안씅밈)

```text
episodes/<episode_name>/
├── metadata.json
├── image/
│   └── frame_*.png
├── pose/
│   └── frame_*.json
└── subgoal/
    └── subgoal_*.json
```

- `metadata.json`의 `map`, `name`, `start`, `goal` 을 평가에 사용함.
- 첫 pose의 `walker.yaw`를 actor 시작 방향으로 사용함.
- pose의 `walker` 위치들을 reference route 길이 계산에 사용함.
- subgoal의 `location`을 policy에 전달할 목표 지점으로 사용함.

에피소드 생성 코드는 잠시 보류


## 9. 평가 결과

평가 종료 후 metric을 terminal에 출력하고 다음 위치에 저장됨.

```text
PythonAPI/benchmark/evaluation/results/
└── <metadata.name>/
    └── <policy>/
        └── metrics.json
```



