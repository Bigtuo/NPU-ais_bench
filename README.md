## 服务器配置
服务器配置如下：
CPU/NPU：鲲鹏 CPU（ARM64）+A300I pro推理卡

系统：Kylin V10 SP1

驱动与固件版本版本：
Ascend-hdk-310p-npu-driver_23.0.1_linux-aarch64.run
Ascend-hdk-310p-npu-firmware_7.1.0.4.220.run
MCU版本：Ascend-hdk-310p-mcu_23.2.3

CANN开发套件：版本7.0.1


## 快速开始
注意：确保驱动固件及CANN已安装！

[驱动固件安装教程](https://blog.csdn.net/weixin_45679938/article/details/142966488)

[CANN安装教程](https://blog.csdn.net/weixin_45679938/article/details/142994060)

[YOLO系列om模型转换测试教程](https://blog.csdn.net/weixin_45679938/article/details/142966255)
```
# 1 创建环境
conda create -n yolo python=3.8.11

# 2 安装基础依赖
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

# 3 安装aclruntime包
pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=aclruntime&subdirectory=ais-bench_workload/tool/ais_bench/backend' -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4 安装ais_bench推理程序包
pip3 install -v 'git+https://gitee.com/ascend/tools.git#egg=ais_bench&subdirectory=ais-bench_workload/tool/ais_bench' -i https://pypi.tuna.tsinghua.edu.cn/simple


# 5 进入运行环境yolo【普通用户】
conda activate yolo
# 6 激活atc【atc --help测试是否可行】
source ~/bashrc

# 7 运行相关测试代码
python YOLO_ais_bench_det_aipp.py
```

## ATC转换命令
YOLOv8s.onnx转换.om参考示例(未带AIPP)：
```
# 转换示例
atc --framework=5 --model=yolov8s.onnx --input_format=NCHW --input_shape="images:1,3,640,640" --output_type=FP32 --output=yolov8s --soc_version=Ascend310P3
```

YOLOv8s.onnx转换.om参考示例(带AIPP)：
```
# 转换示例
atc --framework=5 --model=yolov8s.onnx --input_format=NCHW --input_shape="images:1,3,1280,1280" --output_type=FP32 --output=yolov8s_aipp --soc_version=Ascend310P3  --insert_op_conf=aipp_conf.aippconfig
```