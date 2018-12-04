#!/usr/bin/env python

# Author: Oscar Argueta
# Date: 05/30/2018
# Kache.ai Inc. Copywrite (c) 2018

"""
Script extracts .jpg frames from ROS bags

Usage:
    python bag_to_images.py --bag_dir=~/Desktop/vide-logs/vidlog-31-05-18-101-280-commute/ \
    --save_dir=../frames2 \
    --skip_rate=2 \
    --topic_name="/logging/frame_capture/image_raw"

Flags:
   --bag_dir: path to directory containing ROS bags you wish to process. Must be specified
   --save_dir: path directory you wish to save frames to. Must be specified
   --skip_rate: number of frames to skip. Default is value of 1
   --topic_name: video topic name. Default "/usb_cam/image_raw"
"""

from __future__ import print_function

import os
import sys
import glob
import argparse

import cv2
import rosbag
import rospy

from cv_bridge import CvBridge
from keyboard.msg import Key


KEY_PRESS_LIST = [Key.KEY_x, Key.KEY_SPACE]

CAR_STATE_TOPIC = "/dbw/toyota_dbw/car_state"
IMAGE_STREAM_TOPIC = "/sensors/usb_cam/rgb/image_raw_f/compressed"

class ImageExporter():
    def __init__(self, FLAGS):
        # Dirs
        self.bag_dir = FLAGS.bag_dir
        self.save_dir = FLAGS.save_dir
        # Create save dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()

        self.skip_rate = FLAGS.skip_rate

    def process_frames(self):

        print("Extracting Frames")
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))

        frame_type_count = {key: 0 for key in KEY_PRESS_LIST}

        frame_count = 0
        msg_count = 0
        last_key_press = None
        key_press_msgs = []

        # Iterate through bags
        rosbag_files = sorted(glob.glob(os.path.join(self.bag_dir, "*.bag")))
        for bag_file in rosbag_files:
            # Open bag file. If corrupted skip it
            try:
                with rosbag.Bag(bag_file, 'r') as bag:
                    # Check if desired topics exists in bags
                    recorded_topics = bag.get_type_and_topic_info()[1]
                    if not all(topic in recorded_topics for topic in (CAR_STATE_TOPIC, IMAGE_STREAM_TOPIC)):
                        print("ERROR: Specified topics not in bag file:", bag_file, ". Skipping bag!")
                        continue

                    # Get key presses timings
                    for topic, msg, t in bag.read_messages():
                        # Keep track of "x" and "Space" presses
                        if topic == CAR_STATE_TOPIC:
                            if msg.keypressed in KEY_PRESS_LIST:
                                if last_key_press is None:

                                    last_key_press = msg
                                elif msg.header.stamp.to_sec() - last_key_press.header.stamp.to_sec() > 0.5:

                                    key_press_msgs.append(msg)
                                    last_key_press = msg

                    # Iterate through Image msgs if there are keypresses of interest present
                    if key_press_msgs:
                        print("Extracting Frames from:", bag_file)
                        # Extract frames based on key press timings
                        for topic, msg, t in bag.read_messages():
                            # Add frames to buffer for selection
                            if topic == IMAGE_STREAM_TOPIC:
                                for key_press_msg in key_press_msgs:
                                    if abs(msg.header.stamp.to_sec() - key_press_msg.header.stamp.to_sec()) <= 1 \
                                    and msg_count % self.skip_rate == 0:
                                        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                                        image_name = "frame%04d.jpg" % frame_count
                                        cv2.imwrite(os.path.join(self.save_dir, image_name), cv_image)
                                        frame_count += 1
                                        # update counts
                                        frame_type_count[key_press_msg.keypressed] += 1
                                        break
                            msg_count += 1

                msg_count = 0
                last_key_press = None
                key_presses = []

            except:
                print(bag_file, "Failed!")
                continue


        # Print Summary
        print("Frames Extracted:", frame_count)
        print("Frames from 'x' press:", frame_type_count[Key.KEY_x])
        print("Frames from 'Space' press:", frame_type_count[Key.KEY_SPACE])


if __name__ == '__main__':
     # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',"--bag_dir", help="path to bag files")
    parser.add_argument('-s', "--save_dir", help="path to save extracted frames")
    parser.add_argument('-sr', "--skip_rate", type=int, default=1, help="skip every [x] frames. value of 1 skips no frames")
    FLAGS = parser.parse_args()

    # Verify dirs
    if not os.path.exists(FLAGS.bag_dir):
        print("Directory to bag files does exist")
    elif len(glob.glob(os.path.join(FLAGS.bag_dir, "*.bag"))) < 1 :
        print("No bag files in specified directory")
    else:
        image_exporter = ImageExporter(FLAGS)
        image_exporter.process_frames()
