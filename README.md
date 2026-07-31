# MuJoCo Visual Servo Tracking

一个面向学习和演示的 MuJoCo 视觉伺服项目：从相机画面定位目标，让机械臂持续跟随，也可以完成接触、抓取、抬升和放置。

![Panda 跟随移动目标](mujoco/media/visual-servo-dashboard.gif)

演示使用 Panda、固定 RGB-D 相机、颜色识别和 Hybrid 视觉伺服。红色杯子移动时，末端保持约 `10 cm` 距离并同步跟随。整段 15 秒录制包含 360 次视觉更新，没有丢失目标；稳态 RMS 跟随误差为 `2.2 mm`，P95 为 `3.3 mm`。

[观看高清 MP4](mujoco/media/visual-servo-dashboard.mp4) · [查看原始截图](mujoco/media/visual-servo-dashboard.png)

## 功能

- IBVS、PBVS、Hybrid 三种视觉伺服模式
- RGB-D 位置估计、颜色分割、Grounding DINO + SAM 语义识别
- 位置、速度、力矩和阻抗控制
- 可执行抓取点、双指接触确认、抬升和 pick-and-place
- Panda、FR3、UR、xArm、Kinova、KUKA、Sawyer 和 Unitree G1 等 Menagerie 模型
- 可替换的机械臂 MJCF、目标物体、mesh、夹爪和抓取点
- 原生 MuJoCo viewer 与 16:9 录制面板

## 安装

```bash
git clone --recurse-submodules https://github.com/MengyangGao/visual_servo_tracking.git
cd visual_servo_tracking
conda env create -f environment.yml
conda activate visual_servo
```

如果已经创建过环境：

```bash
conda env update -n visual_servo -f environment.yml --prune
```

语义识别需要额外安装视觉模型依赖：

```bash
python -m pip install -e 'mujoco[semantic]'
```

## 运行演示

macOS 请用 `mjpython` 启动相机渲染和原生 viewer：

```bash
mjpython mujoco/scripts/demo.py \
  --robot panda --target cup --trajectory circle \
  --detector color --servo-mode hybrid
```

拖动鼠标可以自由观察场景。方向键移动目标，`,` 和 `.` 控制目标下降和上升，Space 或 Backspace 清除手动偏移。

使用自然语言寻找目标：

```bash
mjpython mujoco/scripts/demo.py \
  --robot panda --target cup --trajectory circle \
  --detector semantic --prompt "red cup" --servo-mode hybrid
```

执行抓取：

```bash
mjpython mujoco/scripts/demo.py \
  --headless --no-realtime --scripted-target \
  --robot panda --target grasp-cube --trajectory static \
  --detector color --servo-mode pbvs --task grasp --steps 1800
```

执行抓取并放置：

```bash
mjpython mujoco/scripts/demo.py \
  --headless --no-realtime --scripted-target \
  --robot panda --target grasp-cube --trajectory static \
  --detector color --servo-mode pbvs --task pick-place \
  --steps 3200 --place-position 0.42 -0.16 0.25
```

加上 `--record output.mp4` 可以录制状态面板。完整参数见 `mujoco-servo --help`。

Linux 无窗口运行可设置 `MUJOCO_GL=egl`，例如：

```bash
MUJOCO_GL=egl mujoco-servo \
  --headless --no-realtime --scripted-target \
  --robot panda --target cup --trajectory circle \
  --detector color --servo-mode hybrid --steps 1800
```

## 机器人与目标

内置机器人来自锁定版本的 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。

| 机器人 | 用途 |
| --- | --- |
| `panda` | 跟随、接触、双指抓取、放置 |
| `fr3`、`ur5e`、`ur10e` | 跟随、接触 |
| `lite6`、`xarm7` | 跟随；xArm7 使用官方夹爪执行器 |
| `iiwa14`、`kinova-gen3`、`sawyer` | 跟随、接触 |
| `g1-left-arm`、`g1-right-arm` | Unitree G1 固定基座上肢实验 |

自定义目标通过 JSON 描述，可使用基础几何体或 OBJ/STL mesh，并配置质量、摩擦、初始姿态和局部抓取点：

```bash
mujoco-servo --target-file /path/to/targets.json --target my-object
```

自定义机械臂描述符引用 MJCF，并声明关节、执行器、home 位姿、末端 frame 和夹爪：

```bash
mujoco-servo --robot-file /path/to/robots.json --robot my-robot
```

格式和模型来源见 [mujoco/ASSETS.md](mujoco/ASSETS.md)。

## 验证

```bash
conda run -n visual_servo pytest -q mujoco/tests
conda run -n visual_servo ruff check mujoco
conda run -n visual_servo python -m mujoco_servo.benchmark --enforce
```

GitHub Actions 会在 Python 3.10–3.13 上运行测试、覆盖率、wheel 构建和安装后冒烟测试。

## 说明

这是仿真项目，不是可直接部署到真实机械臂的安全控制器。学习视觉可使用 CUDA、Apple MPS 或 CPU；MuJoCo 动力学在 CPU 上运行。
