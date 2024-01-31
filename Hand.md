## python环境配置

- 下载安装[miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
- 打开`Anaconda Prompt`
- 配置环境
```bash
conda create -n hand python=3.10 # 创建 hand 环境，安装python3.10
conda activate hand # 激活环境
pip install mediapipe opencv-python # 安装包
```
- 运行代码 `python .\hand.py`

## 关键点及角度
- 手部关节点信息如下
![](https://developers.google.com/static/mediapipe/images/solutions/hand-landmarks.png)

[参考mediapipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

- 角度计算（以食指为例）
	- angle1: 计算关节点(5,6)与(7,8)夹角
	- angel2: 计算关节点(0,5)与(5,6)夹角
- 分级判断：
	- 1-7分级以angle1角度判断
	- 达到7之后以angle2角度判断是否完全握拳(8级)