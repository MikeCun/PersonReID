# PersonReID
  This is like a kind of engineering project, not a studying one.
  
# Instruction
  
  Firstly, we use Mask R-CNN designed by MatterPort to detect the pedestrian in the frame, then using the EANet designed by Huang Houjing to extract the pedestrian body feature, using the Face Recognition designed by Ageitgey to extract the pedestrian face feature and storing them in a feature database. We define the pedestrian feature as the Pedestrian Unique Feature(PUF). Now we finish the identification about the video, then we can use the image of the pedestrian to re-identification.
  Secondly, we give a querying pedestrian image into the system. When the extracting feature process has finished, we put the PUF of the pedestrian image in the database. The Pedestrian Unique Feature Distinguished Distence(PUFDD) is a distence that we can use it to distinguish two pedestrians if they are one or not and it is also the most important contribution we do in this project. We query the feature database with the feature of querying pedestrian image and use the PUFDD to estimate if it has the same one in the database.
  Finally, we made a web demo system which using the Django to easily use.
  
# Installation

1.Clone this repository

2.Install dependencies
```bash
pip3 install -r requirements.txt
python3 setup.py install
```

Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [MatterPort/Mask_RCNN releases page](https://github.com/matterport/Mask_RCNN/releases).
Download pre-trained EANet model (model_best.tar.pth) from the [Huanghoujing/EANet page](https://drive.google.com/open?id=1pmH8vFYgu_1IFdu-RmK4hYB9J8ft5LjU).

4.(Option) If you want to use the web demo, you need to build a Django web framework. You can use the Pycharm easily to build it up in the localhost, and you also can build it in the Server. Follow the installation, then you input the URL or domain name in the browser and you can see the demo system.

# Getting Started
You will see the main system construction in the [demo.ipynb](https://github.com/MikeCun/PersonReID/blob/master/demo.ipynb).
If you are using the web demo system, the main Rerson ReID service is written in the [service.py](https://github.com/MikeCun/PersonReID/blob/master/service.py).

# Inference
Now the inference is based on the Django Web Framework, we will provide the normal API in the feature.
```python
from service import PersonReID

reid_result = PersonReID(reid_video=reid_video_path, reid_pic=reid_pic_path)
image_path, image_name = reid_result.reid_result()
```
