#!/usr/bin/python3

import os
import tf
import math
import rospy
import random
import subprocess

import torch
import numpy as np
import torch.nn as nn


from std_srvs.srv import Empty
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import GetModelState, SetModelState
from geometry_msgs.msg import Twist

from dqn_controller.msg import ReplayRecord

import models.mapAutoencoder200 as mapAutoencoder
import models.scanAutoencoder360 as scanAutoencoder

EPS_THRESHOLD = 0.05
NUMBER_OF_ACTIONS = 9

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(418, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, NUMBER_OF_ACTIONS),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Controller:
    
    def __init__(self) -> None:
        self.laser_list = []
        self.action_twist = Twist()

        rospy.init_node("controller" , anonymous=True)

        # subscriber
        self.laser_subscriber = rospy.Subscriber("/scan" , LaserScan , callback=self.laser_callback)
        # publisher
        self.cmd_publisher = rospy.Publisher('/cmd_vel' , Twist , queue_size=10)
        self.my_publisher = rospy.Publisher("/data_buffer", ReplayRecord, queue_size=10)


        # models
        self.weight_dir = "/home/asad/catkin_ws2/src/turtlebot3_activeSLAM/dqn_controller/weights/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.map_weight_path = self.weight_dir + sorted([w for w in os.listdir(self.weight_dir) if "map_autoencoder" in w])[-1]
        self.map_autoencoder = mapAutoencoder.Autoencoder().to(self.device)
        self.map_autoencoder.load_state_dict(torch.load(self.map_weight_path))

        self.scan_weight_path = self.weight_dir + sorted([w for w in os.listdir(self.weight_dir) if "scan_autoencoder" in w])[-1]
        self.scan_autoencoder = scanAutoencoder.Autoencoder().to(self.device)
        self.scan_autoencoder.load_state_dict(torch.load(self.scan_weight_path))
        self.update_dqn_network()




    def update_dqn_network(self):
        self.dqn_weight_path = self.weight_dir + sorted([w for w in os.listdir(self.weight_dir) if "dqn" in w])[-1]
        print(self.dqn_weight_path)
        self.dqn_network = DQN().to(self.device)
        self.dqn_network.load_state_dict(torch.load(self.dqn_weight_path))


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



    def take_action(self, map, lasers, twist):
        rand = random.random()

        # for validation phase
        # rand = 1

        if rand > EPS_THRESHOLD:
            with torch.no_grad():
                map = np.array(map.map.data, dtype=np.int8).reshape(-1, 384, 384)
                map = mapAutoencoder.transform_map(map).to(self.device)
                map = torch.stack([self.map_autoencoder.encoder(map)]).reshape(1, -1)
                scan = np.array([l.ranges for l in lasers])
                scan = scanAutoencoder.transform_scan(scan).to(self.device)
                scan = self.scan_autoencoder.encoder(scan).reshape(1, -1)

                velocity = torch.Tensor([twist.linear.x]).to(torch.float32).to(self.device).reshape(1, 1)
                rotation = torch.Tensor([twist.angular.z]).to(torch.float32).to(self.device).reshape(1, 1)

                state = torch.concatenate([map, scan, rotation, velocity], axis=1)
                action = self.dqn_network(state)

                action = action.max(1).indices.item()
        else:
            action = np.random.randint(9)

        assert(0 <= action and action < 9)

        # publish action
        self.action_twist = Twist()
        self.action_twist.linear.x = (action // 3) * 0.15
        self.action_twist.angular.z = (action % 3 - 1) * 0.3
        self.cmd_publisher.publish(self.action_twist)

    
    def run(self):
        NUMBER_OF_STEP = 20
        NUMBER_OF_SCAN = 5

        while not rospy.is_shutdown():
            self.reset_env()
            self.action_twist = Twist()
            self.cmd_publisher.publish(self.action_twist)
            self.laser_list.clear()
            buffer = []

            for i in range(NUMBER_OF_STEP + 1):
                s_map = self.get_map()
                while len(self.laser_list) < NUMBER_OF_SCAN:
                    rospy.sleep((NUMBER_OF_SCAN - len(self.laser_list)) / 100)
                s_laser = self.laser_list[:NUMBER_OF_SCAN]
                self.laser_list = self.laser_list[NUMBER_OF_SCAN:]
                s_twist = self.action_twist

                self.take_action(s_map, s_laser, s_twist)

                print("step:", i)
                buffer.append((s_map, s_laser, s_twist))


            # publish data for learner
            record = ReplayRecord()
            record.maps = [b[0].map for b in buffer]
            record.scans = sum([b[1] for b in buffer], [])
            record.cmds = [b[2] for b in buffer]
            self.my_publisher.publish(record)

            self.update_dqn_network()



if __name__ == "__main__":
    controller = Controller()
    
    controller.run()
