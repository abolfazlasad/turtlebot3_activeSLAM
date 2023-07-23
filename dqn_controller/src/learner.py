#!/usr/bin/python3

import rospy
import numpy as np

from nav_msgs.msg import OccupancyGrid
from dqn_controller.msg import ReplayRecord

class Learner:
    def __init__(self) -> None:
        rospy.init_node("learner", anonymous=True)

        self.my_subscriber = rospy.Subscriber("/data_buffer", ReplayRecord, callback=self.myTopic_callback)


    def myTopic_callback(self, data: ReplayRecord):
        maps = np.array([m.data for m in data.maps], dtype=np.int8).reshape(-1, 384, 384)
        scans = np.array([s.ranges for s in data.scans]).reshape(maps.shape[0], 5, 360)
        velocities = np.array([b.linear.x for b in data.cmds]).reshape(maps.shape[0], 1)
        rotations = np.array([b.angular.z for b in data.cmds]).reshape(maps.shape[0], 1)

        print(maps.shape)
        print(scans.shape)
        print(velocities.shape)
        print(rotations.shape)
        print()


    def run(self):
        while not rospy.is_shutdown():
            rospy.sleep(1)



if __name__ == "__main__":
    learner = Learner()

    learner.run()