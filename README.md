# CV期末大作业
# 1 环境配置
1. Yolo训练环境：   
   - 硬件：RTX 3080 16G Laptop，AMD Ryzen 9 5900HX
   - Windows11 + Anaconda3 + CUDA11.6 +pytorch1.9.1  
2. ViT-SpinalNet训练环境：  
   - 硬件：NVIDIA V100 16G , Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz
   - Ubuntu20.04 + CUDA11.3 + pytorch1.9.1 
# 2 流程
1. 首先在kaggle下载数据集[链接地址](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset)，解压到`datasets`文件夹下，文件结构如下
    ```
    datasets
    |--- classification
    |    |--- test
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |    |--- train
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |    |--- val
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |--- yolo_data
    |    |--- test
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |    |--- train
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |    |--- val
    |    |    |--- 0
    |    |    |--- ...
    |    |    |--- 101
    |--- classes.txt
    |--- test.txt
    |--- train.txt
    |--- val.txt
    ```
2. 按照YOLOv8官方教程训练检测模型，使用的预训练模型为Yolov8-l。训练后模型放置在`yolov8`文件夹下，命名为`best_l.pt`。提供以及微调好的[模型地址](https://pan.baidu.com/s/1-sX5uu21xic1SokRexJDkA?pwd=0000)。
3. 按步骤运行DataPerporcession.ipynb
4. 按步骤运行TrainVitsSpinalNet.ipynb,若想直接查看结果已经提供扩充好的数据集[扩充IP102-IC数据集地址](https://www.kaggle.com/datasets/zbzzcc/myyolo),请将其放置在`datasets/yolo_data`文件夹,以及训练好的ViT-SpinalNet[模型地址](https://pan.baidu.com/s/1-sX5uu21xic1SokRexJDkA?pwd=0000)，请放置在model文件夹下。
5. CBLoss.py为自定义的损失函数的测试文件，构建了一个小数据集进行测试，TranditionalImgCut是传统图像处理的测试文件，使用的是OpenCV的图像处理函数，TranditionalImgCut是传统图像处理的测试文件。
