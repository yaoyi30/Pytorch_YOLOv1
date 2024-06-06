<div align="center">   

# PyTorch YOLOv1 Project
</div>


### 环境配置
python version 3.8, torch 1.8.1, torchvision 0.9.1:
```
pip install torch==1.8.1 torchvision==0.9.1
```


### 数据准备
数据文件夹结构如下:
```
datasets/
  JPEGImages/   # VOC2007 + VOC2012 all images
     img1.jpg
     img2.jpg
       .
       .
       .
       
  train.txt
  val.txt

```
### 训练
```
python train.py --input_size 448 448 --batch_size 32 --epochs 80 --nb_classes 20 --finetune ./weights/resnet50_ram-a26f946b.pth --data_path ./datasets/ --output_dir ./output_dir 
```
### 评价模型
```
python eval.py --input_size 448 448 --weights ./output_dir/best.pth --data_path ./datasets/ --nb_classes 20
```
### 模型预测
```
python predict.py --input_size 448 448 --weights ./output_dir/best.pth --image_path ./person.jpg --nb_classes 20
```
### 导出onnx模型
```
python export_onnx.py --input_size 448 448 --weights ./output_dir/best.pth --nb_classes 20
python -m onnxsim best.onnx best_sim.onnx
```

### 训练过程可视化
#### 1. Loss曲线
![loss.png](output_dir%2Floss.png)
#### 2. 学习率曲线
![learning_rate.png](output_dir%2Flearning_rate.png)


### 结果可视化

#### Train on VOC2007 + VOC2012

| model  | backbone | mAP (VOC2007 test) |
|:------:|:--------------:|:------------------:|
| YOLOv1 |    ResNet50    |        0.65        |

![dog.png](output_dir%2Fdog.png)

![person.png](output_dir%2Fperson.png)

![people.png](output_dir%2Fpeople.png)