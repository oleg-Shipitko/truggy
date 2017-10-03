#include <ros.h>

ros::NodeHandle nh;

void setup()
{
  nh.initNode();
}

void loop()
{
  nh.spinOnce();
}
