import os
import yaml
import urllib
from PIL import Image
from enum import Enum
from pycocotools.coco import COCO

import xml.etree.cElementTree as ET
import glob
import argparse
import numpy as np
import json
import numpy
import cv2
from collections import OrderedDict
import scipy.misc
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
import random
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import shutil
import pickle
import pandas as pd
import subprocess


class Format(Enum):
    scalabel = 0
    coco = 1
    darknet = 2
    bdd = 3
    vgg = 4

##########  ############
##      Refactor      ##
##########  ############

BASE_DIR = '/media/dean/datastore1/datasets/BerkeleyDeepDrive/'
BDD100K_DIRECTORY = os.path.join(BASE_DIR, 'bdd100k')
IMAGE_LIST_DIR = os.path.join(BDD100K_DIRECTORY, 'images/100k/train/image_list.yml')
LABEL_LIST_DIR = os.path.join(BDD100K_DIRECTORY, 'labels/bdd100k_labels_images_train.json')
DEFAULT_IMG_EXTENSION = '.jpg'


class DataFormatter(object):
    def __init__(self, annotations_list, s3_bucket = None, check_s3 = False, data_format=Format.scalabel, output_path=os.getcwd(), pickle_file = None, trainer_prefix = None):
        self._images = {}
        self._annotations = {}
        self.s3_bucket = s3_bucket
        self.check_s3 = check_s3
        self.output_path = output_path
        self.trainer_prefix = trainer_prefix
        self.coco_directory = os.path.join(self.output_path, 'coco')

        # Check if pickle_file is None or does not exist
        if pickle_file and os.path.exists(pickle_file):
            self._images, self._annotations = self.get_cache(pickle_file)
        else:
            path = os.path.normpath(annotations_list)
            self._pickle_file = "{}.pickle".format('_'.join(path.split(os.sep)[5:]))


            ###------------------ Scalabel Data Handler -----------------------###
            if data_format == Format.scalabel:
                with open(image_list, 'r') as stream:
                    image_data = yaml.load(stream)
                    if image_data:
                        for img in image_data:
                            img_url = img['url']
                            fname = os.path.split(img_url)[-1]
                            full_path = self.maybe_download(img_url, img_prefix+fname)
                            if s3_bucket:
                                self.send_to_s3(os.path.join(DARKNET_TRAINING_DIR, img_prefix+fname))

                            im = Image.open(full_path)
                            width, height = im.size
                            self._images[img_prefix+fname] = {'url': img_url, 'coco_path': full_path,
                                                 'width': width, 'height': height}


                # Import Labels
                with open(annotations_list, 'r') as f:
                    data = json.load(f)

                    for ann in data:
                        fname = os.path.split(ann['url'])[-1]
                        self._annotations[img_prefix+fname] = ann['labels']
                        img_data = self._images[img_prefix+fname]
                        img_data['attributes'] = ann['attributes']
                        img_data['videoName'] = ann['videoName']
                        img_data['timestamp'] = ann['timestamp']
                        img_data['index'] = ann['index']

                        self._images[img_prefix+fname] = img_data


            ###------------------ BDD100K Data Handler -----------------------###
            elif data_format == Format.bdd:
                BDD100K_ANNOTATIONS_FILE = os.path.join(self.coco_directory,'annotations/bdd100k_altered_instances_train2014.json')
                BDD100K_VIDEOS_PATH='https://s3-us-west-2.amazonaws.com/kache-scalabel/bdd100k/videos/train/'
                with open(annotations_list, 'r') as f:
                    data = json.load(f)
                    ann_idx = 0
                    for idx, img_label in enumerate(data):
                        img_key = self.trainer_prefix+img_label['name']
                        img_uri = self.maybe_download(os.path.join(BDD100K_DIRECTORY, 'images/100k/train', img_label['name']),
                                                        os.path.join(self.output_path , img_key))

                        if self.s3_bucket: img_uri = self.send_to_s3(img_uri)
                        im = Image.open(img_uri)
                        width, height = im.size

                        if img_label.get('attributes', None):
                            self._images[img_key] = {'url': img_uri, 'name': img_uri, 'coco_path': os.path.join(self.coco_directory, 'images' , self.trainer_prefix.split('_')[1], img_key),
                                                              'width': width, 'height': height, 'labels': [],
                                                              'index': idx, 'timestamp': 10000,
                                                              'videoName': BDD100K_VIDEOS_PATH+"{}.mov".format(os.path.splitext(img_label['name'])[0]),
                                                              'attributes': {'weather': img_label['attributes']['weather'],
                                                                             'scene': img_label['attributes']['scene'],
                                                                             'timeofday': img_label['attributes']['timeofday']}}
                        else:
                            self._images[img_key] = {'url': img_uri, 'name': img_uri, 'coco_path': os.path.join(self.coco_directory, 'images' , self.trainer_prefix.split('_')[1], img_key),
                                                              'width': width, 'height': height, 'labels': [],
                                                              'index': idx, 'timestamp': 10000,
                                                              'videoName': BDD100K_VIDEOS_PATH+"{}.mov".format(os.path.splitext(img_label['name'])[0])}

                        self._annotations[img_key] = []
                        for ann in [l for l in img_label['labels'] if l.get('box2d', None)]:
                            label = {}
                            label['id'] = int(ann_idx)
                            label['attributes'] = ann.get('attributes', None)
                            if ann.get('attributes', None):
                                label['attributes'] = {'Occluded': ann['attributes'].get('occluded', False),
                                                       'Truncated': ann['attributes'].get('truncated', False),
                                                        'Traffic Light Color': [0, 'NA']}

                            label['manual'] =  ann.get('manualShape', True)
                            label['manualAttributes'] = ann.get('manualAttributes', True)
                            label['poly2d'] = ann.get('poly2d', None)
                            label['box3d'] = ann.get('box3d', None)

                            label['box2d'] = {'x1': ann['box2d']['x1'],
                                        'x2': ann['box2d']['x2'],
                                        'y1': ann['box2d']['y1'],
                                        'y2': ann['box2d']['y2']}

                            label['category'] = ann['category']
                            if label['category'] == 'traffic light':
                                if ann['attributes']['trafficLightColor'] == 'green':
                                    label['attributes']['Traffic Light Color'] = [1, 'G']
                                elif ann['attributes']['trafficLightColor'] == 'yellow':
                                    label['attributes']['Traffic Light Color'] = [2, 'Y']
                                elif ann['attributes']['trafficLightColor'] == 'red':
                                    label['attributes']['Traffic Light Color'] = [3, 'R']

                            self._images[img_key]['labels'].append(label)
                            ann_idx +=1
                        self._annotations[img_key].extend(self._images[img_key]['labels'])


            ###------------------ VGG Data Handler-(Legacy Labeler) -----------------------###
            elif data_format == Format.vgg:
                HEADER_ROW=['filename', 'file_size', 'file_attributes', 'region_count',
                            'region_id', 'region_shape_attributes', 'region_attributes']
                vgg_annotations = pd.read_csv(annotations_list, names=HEADER_ROW, skiprows=1)
                img_paths = sorted(set(vgg_annotations['filename'].tolist()))

                num_imgs = len(img_paths)
                ann_idx = 0

                # loop through each image
                urlstofilepaths = {}
                img = {}
                start_idx = int(1e6)
                for idx, img_url in enumerate(img_paths, start=start_idx):
                    img = {}
                    # Download Image if not exist
                    fname = '_'.join(img_url.split('/')[-2:])
                    urlstofilepaths[img_url] = maybe_download(img_url, os.path.join(DARKNET_TRAINING_DIR, img_prefix+fname))
                    # Get Image Size in Bytes
                    img_file_size =  os.stat(urlstofilepaths[img_url]).st_size

                    if s3_bucket:
                        img_url = self.send_to_s3(urlstofilepaths[img_url])


                    img['name'] = img_prefix+fname
                    img['url'] = img_url
                    img['videoName'] = ''
                    img['file_size'] = img_file_size
                    img['index'] = idx
                    img['timestamp'] = 10000
                    img['labels'] = []
                    img['attributes'] = {'weather': 'clear',
                                         'scene': 'highway',
                                         'timeofday': 'night'}
                    self._images[img_prefix+fname] = img
                    self._annotations[img_prefix+fname] = []

                    for annotation in [x for x in vgg_annotations.as_matrix() if x[0].lower() == img_url.lower()]:
                        ann = {}
                        ann['id'] = ann_idx
                        ann['attributes'] = {'Occluded': False, 'Truncated': False}
                        ann['manual'] = True
                        ann['poly2d'] = None
                        ann['box3d'] = None
                        ann['box2d'] = None
                        d = ast.literal_eval(annotation[5])

                        if d:
                            if float(d['x']) < 0.0:
                                d['x'] = 0.0
                            if float(d['y']) < 0.0:
                                d['y'] = 0.0
                            if float(d['height']) <= 0.0:
                                d['height'] = 1.0

                            if float(d['width']) <= 0.0:
                                d['width'] = 1.0

                            ann['box2d'] = {'x1': d['x'],
                                            'x2': d['x'] + d['width'],
                                            'y1': d['y'],
                                            'y2': d['y'] + d['height']}


                        cls = ast.literal_eval(annotation[6])
                        cat = None
                        if cls:
                            cat = cls['type'].lower().strip()
                        if not cat or cat == '' or cat == 'fire hydrant':
                            continue
                        elif cat == 'tlr':
                            ann['attributes']['Traffic Light Color'] = [3, 'R']
                            ann['category'] = 'traffic light'
                        elif cat == 'tlg':
                            ann['attributes']['Traffic Light Color'] = [1, 'G']
                            ann['category'] = 'traffic light'
                        elif cat == 'tla':
                            ann['attributes']['Traffic Light Color'] = [2, 'Y']
                            ann['category'] = 'traffic light'
                        elif cat == 'tlna' or cat == 'traffic light':
                            ann['attributes']['Traffic Light Color'] = [0, 'NA']
                            ann['category'] = 'traffic light'
                        elif cat == 'motorbike':
                            ann['category'] = 'motor bike'
                        elif cat == 'speedlimitsign' or cat == 'stop sign' or cat == 'cone' or cat == 'clock':
                            cat = 'traffic sign'
                        elif cat not in category_names:
                            continue
                        else: # Verify category exists
                            ann['category'] =  ids2cats[cats2ids[cat]]


                        img['labels'].append(ann)
                        ann_idx += 1
                    self._annotations[img_prefix+fname].extend(img['labels'])


            # Save object to picklefile
            pickle_dict = {'images':self._images,'annotations':self._annotations}
            with open(self._pickle_file,"wb") as pickle_out:
                pickle.dump(pickle_dict, pickle_out)

        print(len(self._images))

    def maybe_download(self, source_uri, destination):
        if not os.path.exists(destination):
            if os.path.exists(source_uri):
                #print('Copying file', source_uri, 'to file:', destination)
                shutil.copyfile(source_uri, destination)
            else:
                destination, _ = urllib.request.urlretrieve(source_uri, destination)
                statinfo = os.stat(destination)
        return destination

    def get_cache(self, pickle):
        self._pickle_file = pickle_file
        pickle_in = open(self._pickle_file,"rb")
        pickle_dict = pickle.load(pickle_in)
        return (pickle_dict['images'],pickle_dict['annotations'])

    def send_to_s3(self, img_path):
        s3_path = os.path.join(self.s3_bucket,os.path.split(img_path)[-1])

        if self.check_s3:
            exists = subprocess.call("aws s3 ls {}".format(s3_path))
            if not exists:
                s3_bucket = 's3://'+self.s3_bucket
                res = subprocess.call("aws s3 cp {} {}".format(img_path, s3_bucket))
                print(res)
        return os.path.join('https://s3-us-west-2.amazonaws.com', s3_path)
