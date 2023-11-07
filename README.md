# DETR Description
* object detection model by DETR model, offering a simple finetune and predict demo code.

# Dataset
下载数据并解压
> wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip
    
> unzip balloon_dataset.zip > /dev/null


## Coco dataset converter
下载转化成coco数据集样式的代码
> git clone https://github.com/woctezuma/VIA2COCO

在训练之前，需要将下载好的数据进行一次处理，转化成coco format才能进行训练，如下是demo code；
```
通过sys添加下载的目录，然后调用相关API
import sys
sys.path.append("/kaggle/working/VIA2COCO")
import convert as via2coco

data_path = '/kaggle/working/balloon/'

first_class_index = 0

for keyword in ['train', 'val']:
  input_dir = data_path + keyword + '/'
  input_json = input_dir + 'via_region_data.json'
  categories = ['balloon']
  super_categories = ['N/A']
  output_json = input_dir + 'custom_' + keyword + '.json'
  print('Converting {} from VIA format to COCO format'.format(input_json))
  coco_dict = via2coco.convert(
      imgdir=input_dir,
      annpath=input_json,
      categories=categories,
      super_categories=super_categories,
      output_file_name=output_json,
      first_class_index=first_class_index,
  )
```

# Training
> python train.py

训练23个epoch，可以看到learning rate、average和loss的变化；
```
Epoch: 1/23: 100% 16/16 [00:12<00:00, 1.65it/s, lr=9.36e-5, train average loss=0.324, train loss=0.254]
Epoch: 1/23: 100% 7/7 [00:02<00:00, 3.73it/s, test average loss=0.841, test loss=0.287]
Epoch: 2/23: 100% 16/16 [00:11<00:00, 1.69it/s, lr=9.65e-5, train average loss=0.361, train loss=0.706]
Epoch: 2/23: 100% 7/7 [00:02<00:00, 3.67it/s, test average loss=0.841, test loss=0.287]
Epoch: 3/23: 100% 16/16 [00:11<00:00, 1.74it/s, lr=9.86e-5, train average loss=0.318, train loss=0.166]
Epoch: 3/23: 100% 7/7 [00:02<00:00, 3.86it/s, test average loss=0.841, test loss=0.287]
Epoch: 4/23: 100% 16/16 [00:11<00:00, 1.72it/s, lr=9.97e-5, train average loss=0.345, train loss=0.375]
Epoch: 4/23: 100% 7/7 [00:02<00:00, 3.83it/s, test average loss=0.841, test loss=0.287]
Epoch: 5/23: 100% 16/16 [00:11<00:00, 1.59it/s, lr=0.0001, train average loss=0.311, train loss=0.131]
Epoch: 5/23: 100% 7/7 [00:02<00:00, 3.73it/s, test average loss=0.841, test loss=0.287]
Epoch: 6/23: 100% 16/16 [00:11<00:00, 1.70it/s, lr=9.93e-5, train average loss=0.33, train loss=0.341]
Epoch: 6/23: 100% 7/7 [00:02<00:00, 3.78it/s, test average loss=0.841, test loss=0.287]
Epoch: 7/23: 100% 16/16 [00:11<00:00, 1.59it/s, lr=9.77e-5, train average loss=0.347, train loss=0.166]
Epoch: 7/23: 100% 7/7 [00:02<00:00, 3.83it/s, test average loss=0.841, test loss=0.287]
Epoch: 8/23: 100% 16/16 [00:11<00:00, 1.71it/s, lr=9.52e-5, train average loss=0.369, train loss=0.215]
Epoch: 8/23: 100% 7/7 [00:02<00:00, 3.76it/s, test average loss=0.841, test loss=0.287]
Epoch: 9/23: 100% 16/16 [00:11<00:00, 1.60it/s, lr=9.18e-5, train average loss=0.355, train loss=0.543]
Epoch: 9/23: 100% 7/7 [00:02<00:00, 3.78it/s, test average loss=0.841, test loss=0.287]
Epoch: 10/23: 100% 16/16 [00:11<00:00, 1.64it/s, lr=8.77e-5, train average loss=0.334, train loss=0.293]
Epoch: 10/23: 100% 7/7 [00:02<00:00, 3.73it/s, test average loss=0.841, test loss=0.287]
Epoch: 11/23: 100% 16/16 [00:11<00:00, 1.64it/s, lr=8.29e-5, train average loss=0.312, train loss=0.283]
Epoch: 11/23: 100% 7/7 [00:02<00:00, 3.72it/s, test average loss=0.841, test loss=0.287]
Epoch: 12/23: 100% 16/16 [00:12<00:00, 1.56it/s, lr=7.74e-5, train average loss=0.346, train loss=0.407]
Epoch: 12/23: 100% 7/7 [00:02<00:00, 3.71it/s, test average loss=0.841, test loss=0.287]
Epoch: 13/23: 100% 16/16 [00:12<00:00, 1.64it/s, lr=7.15e-5, train average loss=0.311, train loss=0.256]
Epoch: 13/23: 100% 7/7 [00:02<00:00, 3.71it/s, test average loss=0.841, test loss=0.287]
Epoch: 14/23: 100% 16/16 [00:11<00:00, 1.79it/s, lr=6.51e-5, train average loss=0.345, train loss=0.372]
Epoch: 14/23: 100% 7/7 [00:02<00:00, 3.71it/s, test average loss=0.841, test loss=0.287]
Epoch: 15/23: 100% 16/16 [00:11<00:00, 1.66it/s, lr=5.85e-5, train average loss=0.346, train loss=0.306]
Epoch: 15/23: 100% 7/7 [00:02<00:00, 3.74it/s, test average loss=0.841, test loss=0.287]
Epoch: 16/23: 100% 16/16 [00:11<00:00, 1.78it/s, lr=5.17e-5, train average loss=0.325, train loss=0.113]
Epoch: 16/23: 100% 7/7 [00:02<00:00, 3.74it/s, test average loss=0.841, test loss=0.287]
Epoch: 17/23: 100% 16/16 [00:11<00:00, 1.62it/s, lr=4.49e-5, train average loss=0.322, train loss=0.219]
Epoch: 17/23: 100% 7/7 [00:02<00:00, 3.76it/s, test average loss=0.841, test loss=0.287]
Epoch: 18/23: 100% 16/16 [00:11<00:00, 1.76it/s, lr=3.82e-5, train average loss=0.337, train loss=0.198]
Epoch: 18/23: 100% 7/7 [00:02<00:00, 3.77it/s, test average loss=0.841, test loss=0.287]
Epoch: 19/23: 100% 16/16 [00:12<00:00, 1.72it/s, lr=3.17e-5, train average loss=0.313, train loss=0.376]
Epoch: 19/23: 100% 7/7 [00:02<00:00, 3.71it/s, test average loss=0.841, test loss=0.287]
Epoch: 20/23: 100% 16/16 [00:12<00:00, 1.53it/s, lr=2.55e-5, train average loss=0.313, train loss=0.242]
Epoch: 20/23: 100% 7/7 [00:02<00:00, 3.74it/s, test average loss=0.841, test loss=0.287]
Epoch: 21/23: 100% 16/16 [00:12<00:00, 1.63it/s, lr=1.98e-5, train average loss=0.354, train loss=0.235]
Epoch: 21/23: 100% 7/7 [00:02<00:00, 3.75it/s, test average loss=0.841, test loss=0.287]
Epoch: 22/23: 100% 16/16 [00:12<00:00, 1.53it/s, lr=1.46e-5, train average loss=0.316, train loss=0.0954]
Epoch: 22/23: 100% 7/7 [00:02<00:00, 3.77it/s, test average loss=0.841, test loss=0.287]
Epoch: 23/23: 100% 16/16 [00:12<00:00, 1.64it/s, lr=1.02e-5, train average loss=0.343, train loss=0.211]
Epoch: 23/23: 100% 7/7 [00:02<00:00, 3.78it/s, test average loss=0.841, test loss=0.287]
```


# Predict
使用默认配置进行predict，如果需要更多的val，则需要修改demo code；
> python predict.py

执行完成之后的precision和recall结果如下：
```
IoU metric: bbox
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.742
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.567
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.456
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.730
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.200
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.590
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.456
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.748
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
```


