# 推理环境准备

**python 版本：3.9**

**pytorch**

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**其他依赖**

```bash
# easydict
pip install easydict
# HTTP 服务器
pip install flask requests websockets
# 深度学习算法
pip install numpy opencv-python pillow
# 船舶检测模型
pip install yolov5
pip install dill
# 文字检测、识别
pip install tqdm
# OCR
pip install onnxruntime
# Install ONNX Runtime GPU (CUDA 12.x)
# The default CUDA version for onnxruntime-gpu in pypi is 12.x since 1.19.0
# Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime
# 所以用 1.18.1 版本的
pip install onnxruntime-gpu==1.18.1
```

# python 项目依赖封装为二进制动态库

## 环境

```
Cython>=3.0.3
setuptools>=69.1.0
```

## 步骤

1.将`setup.py`置于项目根目录下

2.修改以下变量为实际路径：

```python
# 编译的c文件保存路径(可以删去)
build_dir = './build_c_dir'
# 项目根目录
arrange_src_dir = './'
# 编译后二进制文件保存路径(打包项目所在路径)
arrange_tgt_dir = '../build_pyd'
# 列举出所有待编译文件所在路径
compile_path = [
    './tracker',
    './keep_detect.py',
    './model.py'
]
```

3.执行命令

```
python setup.py build_ext --inplace
```

4.编译文件(`.so` 或`.pyd`)会生成在`arrange_tgt_dir`路径下，将源项目其余文件迁移到该路径下即可
