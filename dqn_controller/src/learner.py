#!/usr/bin/python3

import time

import rospy

from dqn_controller.msg import ReplayRecord

class Learner:
    def __init__(self) -> None:
        rospy.init_node("learner", anonymous=True)

        self.time = 0

        self.my_subscriber = rospy.Subscriber("/myTopic", ReplayRecord, callback=self.myTopic_callback)


    def myTopic_callback(self, num: ReplayRecord):
        print(num)
        print()


    def run(self):
        while not rospy.is_shutdown():
            # print(time.time() - self.time)
            self.time = time.time()
            rospy.sleep(1)



if __name__ == "__main__":
    learner = Learner()

    learner.run()