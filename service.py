import os
import cv2
import sys
import h5py
import torch
import pickle
import shutil
import argparse
import warnings
import progressbar
import numpy as np
from easydict import EasyDict
from functools import partial

# Import COCO config
from mrcnn import coco
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from package.eval.np_distance import compute_dist
from package.optim.eanet_trainer import EANetTrainer
from face_recognition.api import compare_faces as face_com
from face_recognition.api import face_locations as face_loc
from face_recognition.api import face_encodings as face_enc
from face_recognition.api import load_image_file as face_load

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Mask RCNN Classify Class
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']


# Prepare load model because of encoding method of the EANET model
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
warnings.filterwarnings('ignore')

# Prepare the EANET
eanet_args = EasyDict()
eanet_args.exp_dir = os.path.join(ROOT_DIR, 'imagenet_model')  # EANET Model Path
eanet_args.cfg_file = os.path.join(ROOT_DIR, 'package/config/default.py')
eanet_args.ow_file = os.path.join(ROOT_DIR, 'paper_configs/PCB.txt')
eanet_args.ow_str = "cfg.only_infer = True"

# GPUs can speed the progress of exacting face feature
cuda = torch.cuda.is_available()


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class PersonReID:

    def __init__(self, reid_video, reid_pic):
        """
        :param workmode: Analyze or ReID Query
        :param reid_video: Video Path for Analyze or ReID Query
        :param reid_pic: Picture Path for ReID
        """
        self.mask_rcnn_model = self.mask_rcnn_model_load()
        self.eanet_trainer = EANetTrainer(args=eanet_args)
        self.reid_video_path = reid_video
        self.reid_pic_path = reid_pic
        self.analysis_video()

    def mask_rcnn_model_load(self):
        """
        Load the Mask RCNN Model by MatterPort in COCO DataSet
        :return: Mask RCNN Model
        """
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "imagenet_model/mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        config = InferenceConfig()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        return model

    def analysis_video(self):
        """
        To Analysis the video: Detect the pedestrian in the frame by Mask RCNN designed by MatterPort
                               Exact the feature of body and face respectively by EANet designed by Huang Houjing and Face Recognition designed by Ageitgey
                               Store the feature dict including the pedestrian picture path and feature (np.array)
        :return: None
        """
        # Set the workpath
        workvideo = self.reid_video_path
        videoname = os.path.splitext(os.path.split(workvideo)[1])[0]

        # Load the model
        model = self.mask_rcnn_model
        eanet = self.eanet_trainer

        # Move to videoname file
        newworkpath = os.path.join(os.path.join(ROOT_DIR, 'reid_result'), videoname) + '/'
        if not os.path.exists(newworkpath):
            os.makedirs(os.path.join(newworkpath))

        # Open the video and get the total frame
        capture = cv2.VideoCapture(workvideo)
        total_frame = int(capture.get(7)) + 1

        # Initial the some variable
        p = progressbar.ProgressBar()  # Show the exacting feature progress
        p.start(total_frame)
        is_stopped = True

        # Create the dictionary to store the feature and their path
        body_feature_dict = {'path': [], 'feat': []}
        face_feature_dict = {'path': [], 'feat': []}

        # Default frequence is 1 sec
        fps = int(capture.get(5))
        frame_freq = fps

        while is_stopped:

            # Load the video frame and decide when the video is end by 'is_stopped'
            is_stopped, frame = capture.read()
            frame_now = int(capture.get(1))

            if (frame_now % frame_freq == 0 and is_stopped):

                results = model.detect([frame], verbose=0)  # Detection
                r = results[0]

                for i in range(r['rois'].shape[0]):
                    # Get every pedestrian in the frame
                    if not np.any(r['rois'][i]):
                        continue

                    # Get the pedestrian in the frame and possibility depending the detection result is greater than 0.95
                    if r['class_ids'][i] == 1 and r['scores'][i] >= 0.95:
                        y1, x1, y2, x2 = r['rois'][i]  # Get the target area of pedestrian

                        # Define the pic name of frame using time stamp in video and serial number in the frame detection result
                        pic_name = self.frame_to_time(frame_now, fps=fps) + '_' + str(i)
                        img_path = newworkpath + pic_name + '.jpg'

                        # Store every pedestrian target area
                        if not os.path.exists(img_path):
                            cv2.imwrite(img_path, frame[y1: y2, x1: x2])

                        # Exact the feature of pedestrain body
                        body_feature = eanet.infer_one_im(im_path=img_path, squeeze=True)['feat']
                        body_feature_dict['path'].append(img_path.encode())  # h5py only supports ASCII in string
                        body_feature_dict['feat'].append(body_feature)       # encode() when is stored, decode() when for using

                        # Using the face feature by Face Recognition API
                        face_np = face_load(img_path)
                        # If you don't have the GPUs, you can remove the 'model' params then it will use 'hog' instead of 'cnn'
                        face_location = face_loc(face_np, model="cnn" if cuda else None)
                        if face_location:  # Sometimes can't detect the face in the frame
                            # Exact the face feature of pedestrian
                            face_feature = face_enc(face_np, known_face_locations=face_location)[0]
                            face_feature_dict['path'].append(img_path.encode())  # h5py only supports ASCII in string
                            face_feature_dict['feat'].append(face_feature)       # encode() when is stored, decode() when for using

            p.update(frame_now)  # Update the progress

        # Store the dictionary including the path and feature using h5
        with h5py.File(newworkpath + videoname + "_body_feature_data.h5", "w") as body_feature_file:
            body_feature_file.create_dataset(videoname + '_path_data', data=body_feature_dict['path'])
            body_feature_file.create_dataset(videoname + '_feat_data', data=body_feature_dict['feat'])

        with h5py.File(newworkpath + videoname + "_face_feature_data.h5", "w") as face_feature_file:
            face_feature_file.create_dataset(videoname + '_path_data', data=face_feature_dict['path'])
            face_feature_file.create_dataset(videoname + '_feat_data', data=face_feature_dict['feat'])

        # Close the file and capture
        capture.release()
        p.finish()

    def reid_result(self):
        """
        To make the ReID result of ReID picture in Analyzed Video
        :return: A url list and a url_name list of ReID result including ReID query picture,
                 relative ReID result and picture name
        """
        # Set the ReID picture path
        reidpicpath = self.reid_pic_path
        reid_image = cv2.imread(reidpicpath, cv2.IMREAD_COLOR)

        # Load the model
        model = self.mask_rcnn_model
        eanet = self.eanet_trainer

        # Set the ReID workpath
        reidvideo_path = self.reid_video_path
        videoname = os.path.splitext(os.path.split(reidvideo_path)[1])[0]
        if os.path.join(os.path.join(ROOT_DIR, 'reid_result'), videoname):
            reidworkpath = os.path.join(os.path.join(ROOT_DIR, 'reid_result'), videoname) + '/'
        else:  # If there isn't a feature file of the video
            print("The %s video hasn't exacted the feature!") % (videoname)
            sys.exit(0)

        # Get the ReID picture name
        reidname = os.path.splitext(os.path.split(reidpicpath)[1])[0]

        # Detection
        reid_results = model.detect([reid_image], verbose=0)
        reid_r = reid_results[0]

        # Create the dictionary of ReID result including every detected pedestrian respectively
        reid_dict = {'reid_path': [], 'reid_rank': []}
        reid_body_feature_dict = {'reid_path': [], 'reid_rank': []}
        reid_face_feature_dict = {'reid_path': [], 'reid_rank': []}

        # Open the stored body and face feature respectively in the directory
        with h5py.File(reidworkpath + videoname + "_body_feature_data.h5", "r") as reid_body_feature_file:
            reid_body_feature_dict['reid_path'] = reid_body_feature_file[videoname + '_path_data'].value
            reid_body_feature_dict['reid_feat'] = reid_body_feature_file[videoname + '_feat_data'].value

        with h5py.File(reidworkpath + videoname + "_face_feature_data.h5", "r") as reid_face_feature_file:
            reid_face_feature_dict['reid_path'] = reid_face_feature_file[videoname + '_path_data'].value
            reid_face_feature_dict['reid_feat'] = reid_face_feature_file[videoname + '_feat_data'].value

        for i in range(reid_r['rois'].shape[0]):
            # Progress every pedestrian in the ReID picture
            if not np.any(reid_r['rois'][i]):
                continue

            # Get the pedestrian in the ReID picture and possibility depending the detection result is greater than 0.95
            if reid_r['class_ids'][i] == 1 and reid_r['scores'][i] >= 0.95:
                y1, x1, y2, x2 = reid_r['rois'][i]  # Get the target area of pedestrian

                # Define the pic name of ReID picture using serial number in the detection result
                reid_pic_name = str(reidname) + '_' + str(i)
                reid_img_path = reidworkpath + reid_pic_name + '.jpg'
                reid_temp_list = []  # Using for adding the ReID result

                # Store every pedestrian target area
                if not os.path.exists(reid_img_path):
                    cv2.imwrite(reid_img_path, reid_image[y1: y2, x1: x2])

                # Exact the feature of pedestrain body
                reid_body_feature = eanet.infer_one_im(im_path=reid_img_path, squeeze=False)['feat']

                # Using the cosine distence as indicator to distinguish the different pedestrian
                # cosine_distance is a list of cosine distance among the ReID body feature and body feature which is stored in the directory
                cosine_distance = compute_dist(reid_body_feature, reid_body_feature_dict['reid_feat'])[0]

                # Distinguished distance as a threshold value can distinguish between two pedestrians if they are same one
                distinguished_distance = 2 * np.mean(cosine_distance) - np.max(cosine_distance)

                for j in range(len(cosine_distance)):
                    if cosine_distance[j] <= distinguished_distance:
                        # Add the same pedestrians result by body disinguishing
                        reid_temp_list.append(reid_body_feature_dict['reid_path'][j].decode())

                # Exact the feature of pedestrain face if he has
                reid_face_np = face_load(reid_img_path)
                reid_face_location = face_loc(reid_face_np, model="cnn" if cuda else None)
                if reid_face_location:
                    reid_face_feature = face_enc(reid_face_np, known_face_locations=reid_face_location)[0]
                    # reid_face_result is a list just including 'True' and 'False'
                    reid_face_result = face_com(reid_face_feature_dict['reid_feat'], reid_face_feature, tolerance=0.58)

                    for k in range(len(reid_face_result)):
                        if reid_face_result[k]:
                            # Add the same pedestrians result by face recognition
                            reid_temp_list.append(reid_face_feature_dict['reid_path'][k].decode())

                reid_dict['reid_path'].append(reid_img_path)  # Add the picture path of ReID result

                if len(reid_temp_list) == 0:  # If ReID fails, there is another picture to show the result
                    reid_dict['reid_rank'].append([os.path.join(ROOT_DIR, 'static') + '/fail.jpg'])
                else:
                    reid_dict['reid_rank'].append(list(set(reid_temp_list)))  # 'set' is used for removing repeat items

        # To make the API easily
        url = []  # The transformed url which is in 'static' including the picture name
        url_name = []  # Get the name of picture
        url_new = '/static/query_result/'
        destination_url = os.path.join(ROOT_DIR, 'static/query_result')

        for i in range(len(reid_dict['reid_path'])):

            reid_query_name = os.path.splitext(os.path.split(reid_dict['reid_path'][i])[1])[0]
            url_transform = url_new + reid_query_name + '.jpg'
            url.append(url_transform)
            shutil.copy(reid_dict['reid_path'][i], destination_url)  # Copy the file for the show
            url_name.append(reid_query_name)

            for j in range(len(reid_dict['reid_rank'][i])):
                reid_query_result_name = os.path.splitext(os.path.split(reid_dict['reid_rank'][i][j])[1])[0]
                result_url_transform = url_new + reid_query_result_name + '.jpg'
                url.append(result_url_transform)
                shutil.copy(reid_dict['reid_rank'][i][j], destination_url)  # Copy the file for the show
                url_name.append(reid_query_result_name)

        return url, url_name

    def frame_to_time(self, frame_now, fps):
        """
        Transform the frames into time format (H-M-S-F)
        :param frame_now: The frame of video is playing
        :param fps: Frame Per Second
        :return: Time Stamp likes hour-min-sec-frame (08-20-21-22)
        """
        secs = frame_now // fps
        s, ms = divmod(frame_now, fps)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        time_stamp = ("%02d-%02d-%02d-%03d" % (h, m, s, ms))

        return time_stamp


def main():
    parser = argparse.ArgumentParser(description='PersonReID')
    parser.add_argument('--reid_video', type=str, default='None', help='Video Path for Analyze or ReID')
    parser.add_argument('--reid_pic', type=str, default='None', help='Picture Path for ReID')
    args = parser.parse_args()

    PersonReID_Service = PersonReID(reid_video=args.reid_video, reid_pic=args.reid_pic)

    reid_result_dict = PersonReID_Service.reid_result()

    print(reid_result_dict)


if __name__ == '__main__':
    main()
