# MixTraining
Official codes for our NeurIPS 2021 paper "Bootstrap Your Object Detector via Mixed Training" ([paper](https://proceedings.neurips.cc/paper/2021/file/5e15fb59326e7a9c3d6558ca74621683-Paper.pdf)). 
## Main Results:

|Model|mAP|AP50|AP75|APs|APm|APl|Link|
|----|-------|-----|----|---|---|---|---|
|[mixed_faster_rcnn_swin_small](configs/ours/mixed_faster_rcnn_swin_small_fpn_coco_4x.py)|0.503|0.716| 0.552| 0.347 |0.540| 0.659|[Google](https://drive.google.com/file/d/1dbxJybYigdL8VOm99q7vI7MKHNMDjLBo/view?usp=sharing)|
|[mixed_cascade_rcnn_swin_small](configs/ours/mixed_cascade_rcnn_swin_small_fpn_coco_4x.py)|0.528|0.721|0.580|0.366|0.568|0.686|[Google](https://drive.google.com/file/d/14VVVml9EPqdA1g4vGBnpuI4sWpeqxX3U/view?usp=sharing)|
# Implementation
- ### Enviroment
```
torch==1.6.0
torchvision==0.7.0
wandb==0.10.26
apex==0.1
mmdet==2.11.0
mmcv-full==1.3.0
```
Install required packages with
```
cd ${your_code_dir}
mkdir -p thirdparty
git clone https://github.com/open-mmlab/mmdetection.git thirdparty/mmdetection 
cd thirdparty/mmdetection && git checkout v2.11.0 && python -m pip install -e .
python -m pip install -e .
mkdir -p data 
ln -s ${your_coco_path} data/coco
```
- ### For testing
```shell
bash tools/dist_test.sh ${selected_config} 8
```
where `selected_config` is one of provided script under the `config/bvr` folder.
- ### For training
```shell
bash tools/dist_train.sh ${selected_config} 8
```
where `selected_config` is one of provided script under the `config/bvr` folder.
- ### For more dataset
We have not trained or tested on other dataset. If you would like to use it on other data, please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md).
## Citing us

```
@inproceedings{xu2021bootstrap,
  title={Bootstrap Your Object Detector via Mixed Training},
  author={Xu, Mengde and Zhang, Zheng and Wei, Fangyun and Lin, Yutong and Cao, Yue and Lin, Stephen and Hu, Han and Bai, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
