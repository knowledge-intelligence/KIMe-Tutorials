# Isaac Mission Dispatch

## 1. Download the repository
``` bash
git clone https://github.com/NVIDIA-ISAAC/isaac_mission_dispatch
```

## 2. Run Isaac Mission Dispatch (Deploy with Docker Compose)
``` bash
cd isaac_mission_dispatch/docker_compose
docker compose -f mission_dispatch_services.yaml up
```

## 3. Run AGV Simulator
``` bash
docker run -it --network host nvcr.io/nvidia/isaac/mission-simulator:isaac_ros --robots \
    carter01,4,5 \
    carter02,9,9,3.14,3
```


