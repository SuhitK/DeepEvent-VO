#!/usr/bin/env python

import sys
import argparse
from fnmatch import fnmatchcase

from rosbag import Bag
from tqdm import tqdm

def main():

    ifile = './mvsec_data_bags/outdoor_day2_events.bag'
    write_bag = Bag(ifile, 'w')
    print ('write bag opened')

    ofile = './mvsec_data_bags/outdoor_day2_data.bag'
    read_bag = Bag(ofile, 'r')
    print ('read bag opened')

    iteration = 1
    for topic,msg,t in read_bag:
        if('events' in topic):
            write_bag.write(topic, msg, t)
            iteration = iteration + 1
            print (iteration, topic)

if __name__ == "__main__":
    main()

