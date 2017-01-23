# Finger gaiting
This module provides code for Task 3-2 of the TACMAN project. The robot learns to perform finger gaiting by learning to switch between demonstrated two- and three-finger grips . For more details please read the README file.


## Dependencies
[Allegro hand ROS stack](https://github.com/simlabrobotics/allegro_hand_ros)

[V-REP](http://www.coppeliarobotics.com/)

## File Descriptions
 * matlab/AllegroScript.m 
 
 Main script for learning finger gaiting. The provided demonstrations and settings for the simulation and real hand setup are stored in there. The script sets up the learning and experiment environment used with our policy search toolbox. For learning of the finger gaiting, we use non-parametric REPS algorithm. 
 
 * src/allegro_hand_policy.launch
 
 The ROS launch file for the real hand setup.
 
 * src/allegro_hand_vrep.launch
 
 The ROS launch file for the simulation setup.
 
 * src/allegroVrep.cpp
 
 The ROS node that enables the communication with the V-REP simulator by using its remote API.
 
 * src/controller.cpp
 
 The ROS node that is used to execute the policies provided by the matlab script on the simulation setup and store the recorded data to be used for learning in the matlab script in return.
 
 * src/controllerReal.cpp
 
 The ROS node that is used to execute the policies provided by the matlab script on the real hand and store the recorded data to be used for learning in the matlab script in return.
 
 * src/recordDemonstrations.cpp
 
 The ROS node that is used to record the demonstrations on the real hand setup. It stores the joingt angle and tactile sensor readings into a file.
