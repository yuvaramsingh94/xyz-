# RoboND final project


## 1. Running my pkg

* copy the pkg onto your catkin workspace or use my entire catkin_ws workspace
* source the workpace usually source ~/catkin_ws/devel/setup.bash (might change based on the directory naming)
* Run catkin_make 
* inside the shell_sh folder (roscd udacity_home_service_robot/shell_sh) , one can find all the needed shell script . please edit the map file location path if needed 
* by default it is ``` xterm -e "export TURTLEBOT_GAZEBO_MAP_FILE=~/catkin_ws/src/udacity_home_service_robot/map/final_pro_2.yaml; roslaunch turtlebot_gazebo amcl_demo.launch"```. change it if needed 
* sometime running the shell script might not launch all the nodes .if this occure ,just rerun it one more time 
* it might take upto 5-10 sec to start moving
