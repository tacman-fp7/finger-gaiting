#include "ros/ros.h"
#include "ros/service.h"
#include "ros/service_server.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"

#include "allegro_hand_core_pd/handState.h"
#include "allegro_hand_core_pd/triggerSimulation.h"
#include "biotacs/BT.h"


#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>

#include <Eigen/Dense>

#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#define DOF_JOINTS 16
#define NUM_ELECTRODES 19
using namespace std;

// Topics
#define STATE_TOPIC "/allegroHand/joint_cmd"
#define BIOTAC_TOPIC "/biotacs"
#define LIB_CMD_TOPIC "/allegroHand/lib_cmd"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)


ros::Subscriber state_sub;
ros::Subscriber bt_sub;
ros::Subscriber lib_cmd_sub;


bool record = false;
int recordCounter = 0;
int recordLenght = 50; // timesteps to save
Eigen::MatrixXd recordJoints;
Eigen::MatrixXd recordSensors;
Eigen::VectorXd biotacData;


void btCallback(const biotacs::BT& msg) {

	for(int f = 0; f < 4; f++) {
		biotacData(f*(NUM_ELECTRODES+1)) = msg.BTTared[f].Pdc;

		for(int e = 0; e < NUM_ELECTRODES; e++) {
			biotacData(f*(NUM_ELECTRODES+1)+e+1) = msg.BTTared[f].electrodes[e];
		}
	}
}

// ##############################################################################
// callbacks
void stateCallback(const sensor_msgs::JointState& msg) {

		// save data into matrices
		for(int i = 0; i < DOF_JOINTS; i++) {
			if(record) {
	            recordJoints(recordCounter,i) = msg.position[i];
			}
		}

		for(int i = 0; i < biotacData.size(); i++) {
			if(record) {
                recordSensors(recordCounter,i) = biotacData(i);
			}
		}

		if(record) {
	        recordCounter++;
	        if(recordCounter == recordLenght) {

	            record = false;

	            ofstream myfile;
	            myfile.open("/home/tanneberg/allegro_ws/allegro_hand_ros/demonstration.txt");
	            myfile << recordJoints.colwise().mean() << endl << endl;
	            myfile << recordSensors.colwise().mean() << endl << endl;
	            myfile.close();

	        }
	    }

}


// Called when an external (string) message is received
void libCmdCallback(const std_msgs::String::ConstPtr& msg) {

	string lib_cmd = msg->data.c_str();
	cout << "LIB_CMD: " << lib_cmd << endl;

    if(lib_cmd.compare("record") == 0) {
		cout << "recording.. ";
        recordJoints = Eigen::MatrixXd::Zero(recordLenght, DOF_JOINTS);
        recordSensors = Eigen::MatrixXd::Zero(recordLenght, (NUM_ELECTRODES+1)*4);
        recordCounter = 0;
        record = true;
    }

}

// ##############################################################################
int main(int argc, char** argv) {

	ros::init(argc, argv, "learned_controller_node");
	ros::NodeHandle nh;

	biotacData = Eigen::VectorXd::Zero(4*(NUM_ELECTRODES+1));

	state_sub = nh.subscribe(STATE_TOPIC, 1, stateCallback, ros::TransportHints().tcpNoDelay());
	bt_sub = nh.subscribe(BIOTAC_TOPIC, 1, btCallback, ros::TransportHints().tcpNoDelay());
	lib_cmd_sub = nh.subscribe(LIB_CMD_TOPIC, 1, libCmdCallback, ros::TransportHints().tcpNoDelay());


	while (ros::ok()) {
		usleep(10);
		ros::spinOnce();
	}


	nh.shutdown();

	return 0;
}
