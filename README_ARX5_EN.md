# Installation and Usage Guide (ARX5 + LeRobot)

This document describes the environment setup, USB-CAN configuration, teleoperation, data collection, and real-robot deployment for the ARX5 robotic arm using the LeRobot framework.

---

## 1. Installation & Environment

### 1.1 LeRobot Environment

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
pip install rerun-sdk==0.23.1
```

### 1.2 ARX5 SDK

```bash
git clone https://github.com/real-stanford/arx5-sdk.git
cd arx5-sdk
git checkout ccf276d  # Use an older version; the latest version requires Ubuntu > 20.04

conda env update -n lerobot --file arx5-sdk/conda_environments/py310_environment.yaml
mkdir build && cd build
cmake ..
make -j
```

### 1.3 Other Dependencies

```bash
pip install pyrealsense2  # Intel RealSense SDK
```

---

## 2. ARX5 USB-CAN Setup

Please refer to the ARX5 SDK [documentation](https://github.com/real-stanford/arx5-sdk?tab=readme-ov-file#usb-can-setup)

### 2.1 Device Identification

```bash
ls /dev/ttyACM*
udevadm info -a -n /dev/ttyACM1 | grep serial
udevadm info -a -n /dev/ttyACM2 | grep serial
```

### 2.2 udev Rules

Add the following rules to `/etc/udev/rules.d/arx_can.rules`:

```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="207838963430", SYMLINK+="arxcan0"
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="206D359D4831", SYMLINK+="arxcan1"
```

Reload rules and bring up CAN devices:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up  # Required after each reconnection
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ifconfig can1 up
```

### 2.3 Testing

```bash
# Keyboard teleoperation test
python examples/keyboard_teleop.py X5 can0
python examples/keyboard_teleop.py X5 can1
```

---

## 3. Teleoperation

```bash
python -m src.lerobot.teleoperate \
    --robot.type=ARX5_follower \
    --robot.id=ARX5_follower \
    --teleop.type=ARX5_leader \
    --teleop.id=ARX5_leader \
    --display_data=true
```

### Fixing CXXABI Errors

If you encounter the following error:

```
ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Create a Conda activation script:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
vim $CONDA_PREFIX/etc/conda/activate.d/preload_libstdcxx.sh
```

Add:

```bash
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 4. Data Collection

```bash
python -m Realbot/src/lerobot/record.py \
    --robot.type=ARX5_follower \
    --robot.id=ARX5_follower \
    --teleop.type=ARX5_leader \
    --teleop.id=ARX5_leader \
    --display_data=true \
    --dataset.repo_id=test/x5_test \
    --dataset.num_episodes=2 \
    --dataset.fps=10 \
    --dataset.push_to_hub=false \
    --dataset.single_task="Test ARX5"
```

---

## 5. ARX5 Deployment

We follow the deployment design of [GR00T](https://github.com/NVIDIA/Isaac-GR00T) and [pi0](https://github.com/Physical-Intelligence/openpi), using a **ZeroMQ-based inference framework** to decouple model inference from robot execution.

Only the inference interface needs to be aligned. Please refer to the LeRobot diffusion policy deployment script: `deploy/deploy_for_arx5_dp_act.py`.
