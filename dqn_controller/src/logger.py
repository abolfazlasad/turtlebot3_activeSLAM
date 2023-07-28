#!/usr/bin/python3

import rospy
import datetime
import numpy as np
from pathlib import Path

from dqn_controller.msg import ReplayRecord

class Logger:
    def __init__(self) -> None:
        rospy.init_node("logger", anonymous=True)
        self.my_subscriber = rospy.Subscriber("/data_buffer", ReplayRecord, callback=self.data_buffer_callback)

    def data_buffer_callback(self, data: ReplayRecord):
        maps = np.array([m.data for m in data.maps], dtype=np.int8).reshape(-1, 384, 384)
        scans = np.array([s.ranges for s in data.scans]).reshape(maps.shape[0], 5, 360)
        velocities = np.array([b.linear.x for b in data.cmds]).reshape(maps.shape[0], 1)
        rotations = np.array([b.angular.z for b in data.cmds]).reshape(maps.shape[0], 1)

        # TODO use const variable
        s = "/home/asad/catkin_ws2/src/turtlebot3_activeSLAM/dqn_controller/log/turtlebot3"
        s += "(" + str(maps.shape[0]) + ") "
        s += str(datetime.datetime.now())[:19]
        s += "/"
        Path(s).mkdir(parents=True, exist_ok=True)

        np.save(s + "maps.npy", maps)
        np.save(s + "scans.npy", scans)
        np.save(s + "velocities.npy", velocities)
        np.save(s + "rotations.npy", rotations)


    def run(self):
        while not rospy.is_shutdown():
            rospy.sleep(1)



if __name__ == "__main__":
    logger = Logger()

    logger.run()