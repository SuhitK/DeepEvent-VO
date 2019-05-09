#!/usr/bin/env python

import numpy as np
import rospy

import sys
from dvs_msgs.msg import EventArray, Event
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters

import matplotlib.pyplot as plt
import scipy.misc as smc
from rosbag import Bag

bridge = CvBridge()

message_vals = 0
trajectory_list = []

list_of_topics = ["/davis/left/image_raw", "/visensor/left/image_raw", "/visensor/right/image_raw", "davis/left/pose"]
ifile = "./mvsec_data_bags/outdoor_day2_pose_frames.bag"

o =  Bag(ifile, 'w')
print ("File is opened")

def callback(left_img_msg, left_vi_img_msg, right_vi_img_msg, pose_msg):

    global message_vals
    global ifile
    global o

    message_vals+=1
    print (message_vals)

    # t = pose_msg.header.stamp
    messages = [left_img_msg, left_vi_img_msg, right_vi_img_msg, pose_msg]
    
    for topic,msg_to in zip(list_of_topics, messages):
        t = msg_to.header.stamp
        o.write(topic, msg_to, t)
    

rospy.init_node('event_time_sync', anonymous=True)

left_image_sub = message_filters.Subscriber("/davis/left/image_raw", Image)
left_vi_image_sub = message_filters.Subscriber("/visensor/left/image_raw", Image)
right_vi_image_sub = message_filters.Subscriber("/visensor/right/image_raw", Image)
pose_sub = message_filters.Subscriber("/davis/left/pose", PoseStamped)

ts = message_filters.ApproximateTimeSynchronizer([left_image_sub, left_vi_image_sub, right_vi_image_sub, pose_sub], 10, 0.1, allow_headerless = True)
ts.registerCallback(callback)

rospy.spin()
