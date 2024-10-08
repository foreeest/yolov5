# YOLO-ELLIPSE

此项目拷了yolov5，在它的基础上修改

## 重要代码 ##

code | desciption
---- | --------
detect.py | 原版推理代码
realsense detect.py | 原版 + 自注释
realsense ellipse.py| YOLO + ellipse Pose Estimation
realsense geometry.py| 给ellipse服务的几何计算模块
./utils/dataloader.py| source 参数为666时使用realsense

- `realsense_ellipse_debug.py`及其对应的geometry模块应该功能比较完整(这里的或者xmate control里面实现的是最终版本)
- 其它`realsense_ellipse_*`是一些辅助脚本

### 代码分析


## 运行方法 ##

### realsense ###

**准备：** 先接上一个realsense;切换到当前目录；conda环境为`yolov5`  
**exp6最新，效果较好**

```shell
$ python realsense_detect.py --source 666 --weight runs/train/exp3/weights/best.pt --conf 0.7
```
视频例子
```shell
$ python3 realsense_ellipse.py --source /home/robot/developEye/test_video/0825_2.avi --weight runs/train/exp5/weights/best.pt --conf 0.7  
```

目前调整分辨率须修改dataloader，同时要改geometry来换内参  

### 内窥镜 ###

**准备：** 先接上一个摄像头，不能是realsense;切换到当前目录；conda环境为`yolov5`  

```shell
$ python detect.py --source 0 --weight runs/train/exp3/weights/best.pt --conf 0.7
```
**参数：**source指定摄像头，weight是模型参数，conf是置信度阈值   


## 训练集问题 ##

- 格式
举例来说，要用`real_eye`训练集训练，以此为根目录，`../datasets`中放置名为`real_eye`的文件夹，其中包括训练、验证、测试；
还需要在`./data`中写一份`.yaml`文件，可以参照`real_eye.yaml`;
上述说得简略，详情见[此处](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#23-organize-directories)Option2;
训练运行指令为  
```shell
$ python train.py --img 640 --epochs 100 --data real_eye.yaml --weights yolov5s.pt
```
结果在`./runs`中，运行参照`运行方法`部分  
更多详情参照yolov5官方文档