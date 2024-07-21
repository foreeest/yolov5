# EyePose开发 #

## 运行方法 ##

### realsense ###

**准备：** 先接上一个realsense;切换到当前目录；conda环境为`yolov5`  

```shell
$ python realsense_detect.py --source 666 --weight runs/train/exp3/weights/best.pt --conf 0.7
```

### 内窥镜 ###

**准备：** 先接上一个摄像头，不能是realsense;切换到当前目录；conda环境为`yolov5`  

- `real_eye`训练集约300张的模型
```shell
$ python detect.py --source 0 --weight runs/train/exp3/weights/best.pt --conf 0.7
```
**参数：**source指定摄像头，weight是模型参数，conf是置信度阈值   
**当前效果：**可以较稳定地识别出假眼的**虹膜**为**瞳孔**；较暗的时候预测效果较好；置信度阈值调到0.7基本不会错了  
**训练：** 硬train一发，都是默认参数，且训练集不完全对应    

- `big_real_eye`训练集约为3k张的模型
```shell
$ python detect.py --source 0 --weight runs/train/exp4/weights/best.pt --conf 0.7
```
**当前效果：**效果较差，原因尚未探究  

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

## TODO ##

- [ ] datasets增强
  - [ ] 弄点强弱光照的训练集，只取pupil的来组合成增强训练集
  - [ ] 此文件夹下的`generate_my_datasets.py`  
- [ ] realsense + yolov5
  - [x] step1 yolov5用realsense摄像头
  - [ ] step2 yolov5整合椭圆到/traditional 
- [ ] vision + arm

## 初步测试 ##
*此部分为开发笔记，请忽略*   

**yolov5**  
- 参考资料:
blog https://blog.csdn.net/weixin_42377570/article/details/128675221  
Docs Training  https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/  

### simple demo ###

```shell

$ conda activate yolov5 
$ python detect.py --weights yolov5s.pt --source 0 (普通识别demo)
$ python segment/predict.py --source 0 (普通segmentation)

```

`--conf 0.5`表示只显示推理置信度大于0.5的目标  

**Question**  
1. 应该可以只识别眼睛，别弄些杂七杂八的吧？    
2. 现在保存记录在/run挺占空间的

### train my own ###

- 拿个现有数据集试试先: COCO或者roboflow里的眼睛的     
需要弄成**yolov5格式**来训练   
放在/datasets中，写了一个`real_eye.yaml`  
```shell
$ python train.py --img 640 --epochs 100 --data real_eye.yaml --weights yolov5s.pt
```
不知道为什么图片就说只有61张？哦，没事，是验证集    
运行测试效果  
```shell
$ python detect.py --source 0 --weight runs/train/exp3/weights/best.pt --conf 0.5
```
结果：Iris当pupil，轻微抖动，没有比原来明显提升特别多的感觉  

再练个大一点的数据集  
```shell
$ python train.py --img 640 --epochs 100 --data big_real_eye.yaml --weights yolov5s.pt
```

这个结构**谨参考**，不完全准确，具体参见datasets和yaml  
```
dataset #(数据集名字：例如fire) 
├── images      
       ├── train          
              ├── xx.jpg     
       ├── val         
              ├── xx.jpg 
├── labels      
       ├── train          
              ├── xx.txt     
       ├── val         
              ├── xx.txt
```

- 数据集标注法
用软件  https://app.roboflow.com/ricardo-forrest/yolov5-lviv8/upload  
这软件太卡了  

- 自动化获取**my数据集**
在`generate_my_datasets.py`里搞    
还要加阴影啥的？目前效果可以    
可以用目前的来做自动生成，置信度为0.8基本不会错，远的时候Iris识别成pupil，基本必定；近的时候会分成Iris和Pupil，感觉如果分开的话就取Iris就行了  

- 有哪些已有数据集
roboflow  
哪些地方能搞到更好的训练集  


- 它一个环的意思是用的时候继续收集数据？  
先不管这个

- 现有模型
识别眼球等的，四个框，不知道为啥实时性这么差  https://universe.roboflow.com/cv-project-sfr6h/eye-tracking-j2zz3  
没事先不管这个  

- 有些优化效果的参数，譬如说**非极大值抑制**可以搞  


## demo搭建 ##

先把yolo弄到`pupil_realsense.py`那测试realsense与yolo兼容性  
接着弄到ROS那里  

- 环境问题
在conda yolov5这里搞ros和librealsense      
`conda install -c conda-forge pyrealsense2==2.50.0.3812`  
找不到，直接pip吧,居然already satisfied   
操，detect.py还有点复杂，得看一下    

- 能否挪走detect.py  
考虑把相关文件都挪到项目里  

- detect的框 * 1.2然后从里面找椭圆，涉及图像缩放问题    


- 不知道conda和ROS会有啥问题吗？  


- 没有关闭rs，**感觉没事**     


- 应该不一定要640*480 吗？
一样的代码realsense 480 * 640；内窥镜640 * 480    
然后改成1920 * 640 下面输出会变成384 * 640，这比例也很奇怪  
应该在letterbox里填充为一个步长的倍数了, 360不是stride32的倍数，384是     


- 如果用640 * 480的话需要其内参  

指令
```shell
$ rs-sense-control
```
640 * 480 30hz bgr8  

- realsense_detect跟detect似乎没有区别  

- 注意清理run/detect 不要清理 run/train  
- 有些识别不出椭圆

- 为什么i会变成1？？？

这里面有些奇怪的问题，涉及鲁棒性  
```shell
$ python realsense_ellipse.py --source 666 --weight runs/train/exp3/weights/best.pt --conf 0.7 --nosave
```
这能跑  

roi改大了，椭圆不画了？  
难道roi会影响原图像？难道是引用不是复制？  