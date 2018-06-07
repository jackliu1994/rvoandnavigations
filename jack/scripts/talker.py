#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import String
import numpy as np
import tensorflow as tf
import tensorlayer as tl
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = 1
MAX_GLOBAL_EP = 20000  # 8000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.999
ENTROPY_BETA = 0.005
LR_A = 0.00002  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP



N_S =26
N_A =2
A_BOUND = [np.array([-1.5,-1.5]), np.array([1.5,1.5])]
winname = 'example'
# print(env.unwrapped.hull.position[0])
# exit()
#权重初始化函数
updatetime=0.25
mirrorrate=0.5
deltatime=0.5 #algorithm1中line8 现在的reward+预计的deltatime后的状态的v*γ_   Rcol的计算中也用到了这个参数
gama=0.8

def weight_variable(shape):
    #输出服从截尾正态分布的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class ACNet(object):

    def __init__(self, scope, globalAC=None):
        self.scope = scope
        if scope == GLOBAL_NET_SCOPE:
            ## global network only do inference
            with tf.variable_scope(scope):
                self.Inputself = tf.placeholder(shape=[None, 6], dtype=tf.float32)
                self.Inputobserve1 = tf.placeholder(shape=[None, 8], dtype=tf.float32)
                self.Inputobserve2 = tf.placeholder(shape=[None, 8], dtype=tf.float32)
                self.Inputobserve3 = tf.placeholder(shape=[None, 8], dtype=tf.float32)

                self._build_net()
                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)

                normal_dist = tf.contrib.distributions.Normal(self.mu,np.array([[0.1,0.1]],dtype="float32"))  # for continuous action space

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

        else:
            ## worker network calculate gradient locally, update on global network
            with tf.variable_scope(scope):
                self.Inputself = tf.placeholder(shape=[None, 6], dtype=tf.float32)
                self.Inputobserve1 = tf.placeholder(shape=[None, 8], dtype=tf.float32)
                self.Inputobserve2 = tf.placeholder(shape=[None, 8], dtype=tf.float32)
                self.Inputobserve3 = tf.placeholder(shape=[None, 8], dtype=tf.float32)
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder("float", shape=[None, 1])

                self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):

                    self.mu, self.sigma = self.mu , self.sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(self.mu,self.sigma)  # for continuous action space

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

                with tf.name_scope('local_grad'):
                    self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                    self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)


    def _build_net(self):

        with tf.variable_scope('actor'):  # Policy network
            self.w_red1_a = weight_variable([6, 75])
            self.w_green1_a = weight_variable([8, 75])
            self.w_yellow1_a = weight_variable([6, 50])
            self.w_blue1_a = weight_variable([8, 50])
            self.b_red1_a = bias_variable([75])
            self.b_blue1_a = bias_variable([50])

            self.h_red1_a = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_red1_a) + tf.matmul(self.Inputobserve1, self.w_green1_a) + tf.matmul(
                    self.Inputobserve2, self.w_green1_a) + tf.matmul(self.Inputobserve3, self.w_green1_a) + self.b_red1_a)
            self.h_blue1_a = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1_a) + tf.matmul(self.Inputobserve1, self.w_blue1_a) + self.b_blue1_a)
            self.h_blue2_a = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1_a) + tf.matmul(self.Inputobserve2, self.w_blue1_a) + self.b_blue1_a)
            self.h_blue3_a = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1_a) + tf.matmul(self.Inputobserve3, self.w_blue1_a) + self.b_blue1_a)

            self.w_red2_a = weight_variable([75, 75])
            self.w_green2_a = weight_variable([50, 75])
            self.w_yellow2_a = weight_variable([75, 50])
            self.w_blue2_a = weight_variable([50, 50])
            self.b_red2_a = bias_variable([75])
            self.b_blue2_a = bias_variable([50])

            self.h_red2_a = tf.nn.relu(
                tf.matmul(self.h_red1_a, self.w_red2_a) + tf.matmul(self.h_blue1_a, self.w_green2_a) + tf.matmul(
                    self.h_blue2_a, self.w_green2_a) + tf.matmul(self.h_blue3_a, self.w_green2_a) + self.b_red2_a)
            self.h_blue4_a = tf.nn.relu(
                tf.matmul(self.h_red1_a, self.w_yellow2_a) + tf.matmul(self.h_blue1_a, self.w_blue2_a) + self.b_blue2_a)
            self.h_blue5_a = tf.nn.relu(
                tf.matmul(self.h_red1_a, self.w_yellow2_a) + tf.matmul(self.h_blue2_a, self.w_blue2_a) + self.b_blue2_a)
            self.h_blue6_a = tf.nn.relu(
                tf.matmul(self.h_red1_a, self.w_yellow2_a) + tf.matmul(self.h_blue3_a, self.w_blue2_a) + self.b_blue2_a)

            self.concated_a = tf.concat([tf.reshape(self.h_blue4_a, [-1, 50, 1]), tf.reshape(self.h_blue5_a, [-1, 50, 1]),tf.reshape(self.h_blue6_a, [-1, 50, 1])], 2)
            self.maxbottom_a = tf.reduce_max(self.concated_a, reduction_indices=[2], keep_dims=True)

            self.w_fc1_a = weight_variable([125, 75])
            self.b_fc1_a = bias_variable([75])
            self.maxall_a = tf.concat([tf.reshape(self.maxbottom_a, [-1, 50]), self.h_red2_a], 1)
            self.h_fc1_a= tf.matmul(self.maxall_a, self.w_fc1_a) + self.b_fc1_a

            self.w_fc2_a = weight_variable([75, 25])
            self.b_fc2_a = bias_variable([25])
            self.h_fc2_a = tf.matmul(self.h_fc1_a, self.w_fc2_a) + self.b_fc2_a

            self.w_fc3_a = weight_variable([25, 2])
            self.b_fc3_a = bias_variable([2])
            self.mu = tf.tanh(tf.matmul(self.h_fc2_a, self.w_fc3_a) + self.b_fc3_a)
            # self.mu = tf.nn.relu6(tf.matmul(self.h_fc2_a, self.w_fc3_a) + self.b_fc3_a)/3.0-1.0

            self.w_fc2_a_1 = weight_variable([75, 25])
            self.b_fc2_a_1 = bias_variable([25])
            self.h_fc2_a_1 = tf.matmul(self.h_fc1_a, self.w_fc2_a_1) + self.b_fc2_a_1

            self.w_fc3_a_1 = weight_variable([25, 2])
            self.b_fc3_a_1 = bias_variable([2])
            self.sigma = tf.nn.softplus(tf.matmul(self.h_fc2_a_1, self.w_fc3_a_1) + self.b_fc3_a_1)

        with tf.variable_scope('critic'):  # we use Value-function here, but not Q-function.
            self.w_red1 = weight_variable([6, 75])
            self.w_green1 = weight_variable([8, 75])
            self.w_yellow1 = weight_variable([6, 50])
            self.w_blue1 = weight_variable([8, 50])
            self.b_red1 = bias_variable([75])
            self.b_blue1 = bias_variable([50])

            self.h_red1 = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_red1) + tf.matmul(self.Inputobserve1, self.w_green1) + tf.matmul(
                    self.Inputobserve2, self.w_green1) + tf.matmul(self.Inputobserve3, self.w_green1) + self.b_red1)
            self.h_blue1 = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1) + tf.matmul(self.Inputobserve1, self.w_blue1) + self.b_blue1)
            self.h_blue2 = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1) + tf.matmul(self.Inputobserve2, self.w_blue1) + self.b_blue1)
            self.h_blue3 = tf.nn.relu(
                tf.matmul(self.Inputself, self.w_yellow1) + tf.matmul(self.Inputobserve3, self.w_blue1) + self.b_blue1)

            self.w_red2 = weight_variable([75, 75])
            self.w_green2 = weight_variable([50, 75])
            self.w_yellow2 = weight_variable([75, 50])
            self.w_blue2 = weight_variable([50, 50])
            self.b_red2 = bias_variable([75])
            self.b_blue2 = bias_variable([50])

            self.h_red2 = tf.nn.relu(
                tf.matmul(self.h_red1, self.w_red2) + tf.matmul(self.h_blue1, self.w_green2) + tf.matmul(
                    self.h_blue2, self.w_green2) + tf.matmul(self.h_blue3, self.w_green2) + self.b_red2)
            self.h_blue4 = tf.nn.relu(
                tf.matmul(self.h_red1, self.w_yellow2) + tf.matmul(self.h_blue1, self.w_blue2) + self.b_blue2)
            self.h_blue5 = tf.nn.relu(
                tf.matmul(self.h_red1, self.w_yellow2) + tf.matmul(self.h_blue2, self.w_blue2) + self.b_blue2)
            self.h_blue6 = tf.nn.relu(
                tf.matmul(self.h_red1, self.w_yellow2) + tf.matmul(self.h_blue3, self.w_blue2) + self.b_blue2)

            self.concated = tf.concat([tf.reshape(self.h_blue4, [-1, 50, 1]), tf.reshape(self.h_blue5, [-1, 50, 1]),
                                       tf.reshape(self.h_blue6, [-1, 50, 1])], 2)
            self.maxbottom = tf.reduce_max(self.concated, reduction_indices=[2], keep_dims=True)

            self.w_fc1 = weight_variable([125, 75])
            self.b_fc1 = bias_variable([75])
            self.maxall = tf.concat([tf.reshape(self.maxbottom, [-1, 50]), self.h_red2], 1)
            self.h_fc1 = tf.matmul(self.maxall, self.w_fc1) + self.b_fc1

            self.w_fc2 = weight_variable([75, 25])
            self.b_fc2 = bias_variable([25])
            self.h_fc2 = tf.matmul(self.h_fc1, self.w_fc2) + self.b_fc2

            self.w_fc3 = weight_variable([25, 1])
            self.b_fc3 = bias_variable([1])
            self.v = tf.tanh(tf.matmul(self.h_fc2, self.w_fc3) + self.b_fc3)*2
            # self.v = tf.nn.relu6(tf.matmul(self.h_fc2, self.w_fc3) + self.b_fc3)/2.0-1.0
    def update_global(self,statevalue0,updateaction,updatevalue): # run by a local

       sess.run([self.update_a_op, self.update_c_op], feed_dict={self.Inputself: statevalue0[:,0:6], self.Inputobserve1: statevalue0[:,6:14],self.Inputobserve2: statevalue0[:,14:22], self.Inputobserve3: statevalue0[:,22:30],self.a_his: updateaction,self.v_target: updatevalue})  # local grads applies to global net
        # return sess.run(self.A,
        #                 feed_dict={self.Inputself: statevalue0[:, 0:6], self.Inputobserve1: statevalue0[:, 6:14],
        #                            self.Inputobserve2: statevalue0[:, 14:22],
        #                            self.Inputobserve3: statevalue0[:, 22:30]})[0]

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, statevalue0):  # run by a local

        return sess.run(self.mu,
                        feed_dict={self.Inputself: statevalue0[:,0:6], self.Inputobserve1: statevalue0[:,6:14],
                                   self.Inputobserve2: statevalue0[:,14:22], self.Inputobserve3: statevalue0[:,22:30]})[0]

    def get_sigma(self, statevalue0):  # run by a local

        return sess.run(self.sigma,
                        feed_dict={self.Inputself: statevalue0[:,0:6], self.Inputobserve1: statevalue0[:,6:14],
                                   self.Inputobserve2: statevalue0[:,14:22], self.Inputobserve3: statevalue0[:,22:30]})[0]


    def get_value(self, statevalue0):  # run by a local

        return sess.run(self.v,
                        feed_dict={self.Inputself: statevalue0[:, 0:6], self.Inputobserve1: statevalue0[:, 6:14],
                                   self.Inputobserve2: statevalue0[:, 14:22],self.Inputobserve3: statevalue0[:, 22:30]})[0]


    def save_ckpt(self,modelname):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model_'+modelname+'.ckpt', var_list=self.a_params + self.c_params, save_dir=self.scope, printable=True)

    def load_ckpt(self,dir,name):
        tl.files.load_ckpt(sess=sess, mode_name=name, var_list=self.a_params+self.c_params, save_dir=dir, is_latest=False, printable=True)



sess = tf.Session()
def talker():

    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
    GLOBAL_AC.load_ckpt(dir="/media/jackliu/e/a3c/fixed_Global_Net_0.00002_0.0001--noinit-entro0-withcollidebreak-boundcorrected",name="model_12000")
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    statusnow_robot_change = np.ones([1,30])
    action_selected_robot = GLOBAL_AC.choose_action(statusnow_robot_change)
    while not rospy.is_shutdown():
        hello_str = "hello world jack!!!!!!!!! %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        rospy.loginfo(action_selected_robot)
     
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
