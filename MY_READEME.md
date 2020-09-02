# Installation
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y

# install the latest mmcv
pip install mmcv-full

# install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

# Download pre-trained model
Faster R-CNN (backbone: ResNet 101) + FPN:
```
mkdir checkpoints
wget -c https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth \
      -O checkpoints/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth
```

# Run
```
python demo/custom_prediction.py -ddir <path/to/PoseTrack/directory> \
    -cfg configs/faster_rcnn/faster_rcnn_r101_caffe_fpn_1x_coco.py \
    -ckpt checkpoints/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth
```
