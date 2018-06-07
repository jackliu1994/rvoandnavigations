/*********************************************************************************
**Fast Odometry and Scene Flow from RGB-D Cameras based on Geometric Clustering	**
**------------------------------------------------------------------------------**
**																				**
**	Copyright(c) 2017, Mariano Jaimez Tarifa, University of Malaga & TU Munich	**
**	Copyright(c) 2017, Christian Kerl, TU Munich								**
**	Copyright(c) 2017, MAPIR group, University of Malaga						**
**	Copyright(c) 2017, Computer Vision group, TU Munich							**
**																				**
**  This program is free software: you can redistribute it and/or modify		**
**  it under the terms of the GNU General Public License (version 3) as			**
**	published by the Free Software Foundation.									**
**																				**
**  This program is distributed in the hope that it will be useful, but			**
**	WITHOUT ANY WARRANTY; without even the implied warranty of					**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the				**
**  GNU General Public License for more details.								**
**																				**
**  You should have received a copy of the GNU General Public License			**
**  along with this program. If not, see <http://www.gnu.org/licenses/>.		**
**																				**
*********************************************************************************/

#include <stdio.h>

#include "ros/ros.h"  
#include "std_msgs/String.h" 
#include "sensor_msgs/Image.h" 
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


int main(int argc ,char **argv)
{	


    //初始化3D显示场景
   // vl.initializeSceneDatasets();
    ros::init(argc, argv, "VO_SF_Datasets");//初始化ROS系统

    ros::start(); //ROS启动函数

    ros::NodeHandle nh;  //建立ROS节点

    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/water_astra_rgbd/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/water_astra_rgbd/depth_registered/image_raw", 1);
 
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
  message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);

    //sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,_1,_2));

    ros::spin();

    return 0;
}
