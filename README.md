## Environment configuration.

[pytorch official](https://pytorch.org/)

python解释器不支持3.8以下的

```bash
conda create -n p2pnet38 python=3.8 -y
conda activate p2pnet38
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt

pip install scikit-learn
```

## Dataset creation.

在xml文件中直接做成点的信息的，直接转换成 x y format的文件

```bash
python crowd_datasets/dataset_tools/convert_to_center_points.py
```

检测一下转化的有没有问题 可视化

```bash
python crowd_datasets/dataset_tools/convert_to_center_points.py
```

生成了对应比例的train.txt  val.test  test.txt (jpg txt放在同一个文件夹中)

```bash
python crowd_datasets/split_jpg_to_txt.py
```

将已将处理好的 image 和 txt文件移动到对应的位置  分别为train  test文件夹

```bash
python crowd_datasets/move_files_by_txt.py
```

下面就是按照数据集训练对应的格式生成对应的.list文件

```bash
python crowd_datasets/dataset_tools/create_list.py
```

[Crowd Counting P2PNet 复现](https://blog.csdn.net/zqq19980906_/article/details/125656654)

## Train

```bash
python train.py
```

对log里面的指标进行训练过程中可视化处理：

```bash
python  visualize_training_metrics.py
python tools_usr/vision_mae_mse.py  # 功能同上
```



## Test

对单个图片进行验证

```bash
python run_test.py --img_path "crowd_datasets/SHHA/test/img_0042_726071616.jpg" --weight_path ckpt/best_mae.pth 
```

对文件进行检测 

```bash
python run_test_processfor_folder.py  --test_data "crowd_datasets/SHHA/test" --output_dir 'output/pre_result'  --weight_path ckpt/best_mae.pth
```

一定要注意,这种插值方法会出问题

```python
        # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        img_raw = img_raw.resize((new_width, new_height))
```

要确保// 128 * 128，确定是整数，

```python
new_width = width // 128 * 128
new_height = height // 128 * 128
```

不然后面经过处理，会导致维度不匹配

```python
        P4_x = self.P4_1(C4)  # P4_x--->torch.Size([1, 256, 189, 240])
        P4_x = P5_upsampled_x + P4_x  # P5_upsampled_x--->torch.Size([1, 256, 188, 240])
```

## metrics

```bash
python  tools_usr/calculate_metric.py
```

