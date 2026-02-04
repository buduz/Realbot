# 安装与使用说明（ARX5 + LeRobot）

本文档整理了 ARX5 真实机械臂在 LeRobot 框架下的环境配置、USB-CAN 设置、遥操作、数据采集以及真机部署流程。

---

## 1. Installation & Environment

### 1.1 LeRobot 环境

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
git checkout ccf276d

conda env update -n lerobot --file arx5-sdk/conda_environments/py310_environment.yaml
mkdir build && cd build
cmake ..
make -j
```

### 1.3 Other Dependencies

```bash
pip install pyrealsense2
```

---

## 2. ARX5 USB-CAN Setup

参考：https://github.com/real-stanford/arx5-sdk?tab=readme-ov-file#usb-can-setup

### 2.1 设备识别

```bash
ls /dev/ttyACM*
udevadm info -a -n /dev/ttyACM1 | grep serial
udevadm info -a -n /dev/ttyACM2 | grep serial
```

### 2.2 udev 规则

```bash
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="207838963430", SYMLINK+="arxcan0"
SUBSYSTEM=="tty", ATTRS{idVendor}=="16d0", ATTRS{idProduct}=="117e", ATTRS{serial}=="206D359D4831", SYMLINK+="arxcan1"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up #（此行每次连接后都必须运行）
sudo slcand -o -f -s8 /dev/arxcan1 can1 && sudo ifconfig can1 up
```

### 2.3 测试

```bash
# 键盘遥操作测试
python examples/keyboard_teleop.py X5 can0
python examples/keyboard_teleop.py X5 can1
```

---

## 3. 傀儡臂遥操作

```bash
python -m src.lerobot.teleoperate \
    --robot.type=ARX5_follower \
    --robot.id=ARX5_follower \
    --teleop.type=ARX5_leader \
    --teleop.id=ARX5_leader \
    --display_data=true
```

### CXXABI 错误修复

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 4. 数据采集

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

## 5. ARX5部署

参考[GR00T](https://github.com/NVIDIA/Isaac-GR00T) and  [pi0](https://github.com/Physical-Intelligence/openpi),使用**ZeroMQ-based inference framework** 解耦模型推理和机械臂动作执行. 只需要对齐模型推理的接口, 参考diffusion policy(lerobot版本)的部署代码deploy/deploy_for_arx5_dp_act.py.
