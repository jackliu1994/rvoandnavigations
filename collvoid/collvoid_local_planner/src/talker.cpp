#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseStamped.h"
#include <sstream>

/**
 * This tutorial demonstrates simple sending of messages over the ROS system.
 */
int main(int argc, char **argv)
{

  ros::init(argc, argv, "talker");


  ros::NodeHandle n;


  ros::Publisher robot_pub1 = n.advertise<geometry_msgs::PoseStamped>("/robot_1/move_base_simple/goal", 1);
  ros::Publisher robot_pub0 = n.advertise<geometry_msgs::PoseStamped>("/robot_0/move_base_simple/goal", 1);
  ros::Rate loop_rate(1);

 

  while (ros::ok())
  {
 
    geometry_msgs::PoseStamped msg;
 
    msg.header.stamp=ros::Time::now();
    ROS_INFO("xixixi");
    msg.header.frame_id="map";
    msg.pose.position.x=-1.348;
    msg.pose.position.y=1.0;
    msg.pose.position.z=0.0;
    msg.pose.orientation.w=1.0;

    robot_pub1.publish(msg);

    msg.pose.position.x=-1.358;
    msg.pose.position.y=3.0;
    msg.pose.position.z=0.0;
    msg.pose.orientation.w=1.0;


    robot_pub0.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();
 
  }


  return 0;
}
