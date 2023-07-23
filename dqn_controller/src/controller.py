#!/usr/bin/python3

import tf
import rospy
import subprocess

import numpy as np


from std_srvs.srv import Empty
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist

from dqn_controller.msg import ReplayRecord



class Controller:
    
    def __init__(self) -> None:
        self.laser_list = []
        self.action_twist = Twist()

        rospy.init_node("controller" , anonymous=True)

        # subscriber
        self.laser_subscriber = rospy.Subscriber("/scan" , LaserScan , callback=self.laser_callback)
        # publisher
        self.cmd_publisher = rospy.Publisher('/cmd_vel' , Twist , queue_size=10)
        self.my_publisher = rospy.Publisher("/myTopic", ReplayRecord, queue_size=10)


        self.get_map()
        # TODO getting specified parameters
        # self.linear_speed = rospy.get_param("/controller/linear_speed") # m/s

        # defining the states of our robot
        # self.GO, self.ROTATE = 0, 1
        # self.state = self.GO 


    def get_map(self):
        rospy.wait_for_service("/dynamic_map")
        service = rospy.ServiceProxy("/dynamic_map", GetMap)
        return service()

    def reset_env(self):
        rospy.wait_for_service("/gazebo/set_model_state")
        service = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        state = ModelState()
        state.model_name ="turtlebot3_waffle"
        x, y = 0, 0
        while min([(x-i)**2 + (j - y)**2 for i in [-1.1, 0, 1.1] for j in [-1.1, 0, 1.1]]) < 0.3:
            x = np.random.random() * 3 - 1.5
            y = np.random.random() * 3 - 1.5

        state.pose.position.x = x
        state.pose.position.y = y
        yaw = np.random.random() * np.pi * 2

        state.pose.orientation.x, \
            state.pose.orientation.y, \
                state.pose.orientation.z, \
                    state.pose.orientation.w, = tf.transformations.quaternion_from_euler(0, 0, yaw)
        service(state)

        subprocess.call(["rosnode", "kill", "/turtlebot3_slam_gmapping"])
        rospy.sleep(1)




    def laser_callback(self, msg: LaserScan):
        self.laser_list.append(msg)

        # TODO use for collision
        # if msg.ranges[0] <= self.stop_distance:
            # self.state = self.ROTATE

    def take_action(self):
        action = np.random.randint(9)
        # publish action
        self.action_twist = Twist()
        self.action_twist.linear.x = (action / 3 - 1) * 0.2
        self.action_twist.angular.z = (action % 3 - 1) * 0.3
        self.cmd_publisher.publish(self.action_twist)

    
    def run(self):
        NUMBER_OF_STEP = 20
        NUMBER_OF_EPISODE = 5

        NUMBER_OF_SCAN = 5

        remain_episode = NUMBER_OF_EPISODE


        buffer = []

        while remain_episode > 0 and not rospy.is_shutdown():
            remain_episode -= 1

            self.reset_env()
            self.action_twist = Twist()
            self.laser_list.clear()

            for i in range(NUMBER_OF_STEP + 1):
                s_map = self.get_map()
                while len(self.laser_list) < NUMBER_OF_SCAN:
                    rospy.sleep((NUMBER_OF_SCAN - len(self.laser_list)) / 20)
                s_laser = self.laser_list[:NUMBER_OF_SCAN]
                self.laser_list = self.laser_list[NUMBER_OF_SCAN:]
                s_twist = self.action_twist

                self.take_action()

                print("step:", i)
                buffer.append((s_map, s_laser, s_twist))


            # publish data for learner
            # record = ReplayRecord()
            # record.data = len(self.laser_list), 10
            # self.my_publisher.publish(record)
            #####



if __name__ == "__main__":
    controller = Controller()
    
    controller.run()
