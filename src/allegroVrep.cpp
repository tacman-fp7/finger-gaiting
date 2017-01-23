#include <iostream>
#include <fstream>

#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/thread/locks.hpp>
#include <boost/scoped_ptr.hpp>

#include "ros/ros.h"
#include "ros/service.h"
#include "ros/service_server.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"

#include "allegro_hand_core_pd/triggerSimulation.h"
#include "allegro_hand_core_pd/getSensorDirections.h"
#include "allegro_hand_core_pd/handState.h"

#include <stdio.h>
#include <math.h>
#include <string>

#include <kdl_parser/kdl_parser.hpp>

#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>

#include "allegro_hand_core_cart/setWeightMatrix.h"
#include "allegro_hand_core_cart/setRegularization.h"

#include <Eigen/Dense>

extern "C" {
    #include "extApi.h"
    #include "extApiPlatform.h"
    #include "v_repConst.h"
}

using namespace std;

// ###############################################################################################################################
// ROS variables
// ###############################################################################################################################
// Topics
// outputs
#define STATE_TOPIC "/allegroHand/state"

// inputs
#define JOINT_CMD_TOPIC "/allegroHand/joint_cmd"
#define TASK_CMD_TOPIC "/allegroHand/task_cmd"
#define LIB_CMD_TOPIC "/allegroHand/lib_cmd"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

// state publisher
ros::Publisher state_publisher;

// service for sensor directions
ros::ServiceServer serviceSensorDirections;

// subscribers for joint, task and string input
ros::Subscriber task_cmd_sub;
ros::Subscriber joint_cmd_sub;		// handles external joint command (eg. sensor_msgs/JointState)
ros::Subscriber lib_cmd_sub;	// handles any other type of eternal command (eg. std_msgs/String)

// messages
allegro_hand_core_pd::handState handStateMessage;
string lib_cmd;


// ###############################################################################################################################
// hand variables
// ###############################################################################################################################
#define DOF_JOINTS 16
double k_p[DOF_JOINTS] = {
	1.0, 1.0, 1.0, 1.0,  // Default P Gains for PD Controller
	1.0, 1.0, 1.0, 1.0,	// These gains are loaded if the 'gains_pd.yaml' file is not loaded
	1.0, 1.0, 1.0, 1.0,
	1.0, 1.0, 1.0, 1.0
};


double k_d[DOF_JOINTS] =
{
	1.0,   1.0,   1.0,   1.0,  // Default D Gains for PD Controller
	1.0,   1.0,   1.0,   1.0,	// These gains are loaded if the 'gains_pd.yaml' file is not loaded
	1.0,   1.0,   1.0,   1.0,
	1.0,   1.0,   1.0,   1.0
} ;

double home_pose[DOF_JOINTS] =
{
	0.0,  -10.0,   45.0,   45.0,  // Default (HOME) position (degrees)
	0.0,  -10.0,   45.0,   45.0,	// This position is loaded and set upon system start
	5.0,   -5.0,   50.0,   45.0,	// if no 'initial_position.yaml' parameter is loaded.
	105.0,  -10.0,   5.0,   10.0
};


string jointNames[DOF_JOINTS] 	=
{
    "joint_0_0",    "joint_1_0",    "joint_2_0",   "joint_3_0" ,
	"joint_4_0",    "joint_5_0",    "joint_6_0",   "joint_7_0" ,
	"joint_8_0",    "joint_9_0",    "joint_10_0",  "joint_11_0",
	"joint_12_0",   "joint_13_0",   "joint_14_0",  "joint_15_0"
};

string sensorNames[4] =
{
    "3_tactile_point_forces", "7_tactile_point_forces", "11_tactile_point_forces" , "15_tactile_point_forces"
};


// joint state variables
float current_position[DOF_JOINTS] 			= {0.0};
double previous_position[DOF_JOINTS]			= {0.0};

double current_position_filtered[DOF_JOINTS] 	= {0.0};
double previous_position_filtered[DOF_JOINTS]	= {0.0};

double current_velocity[DOF_JOINTS] 			= {0.0};
double previous_velocity[DOF_JOINTS] 			= {0.0};
double current_velocity_filtered[DOF_JOINTS] 	= {0.0};

double desired_position[DOF_JOINTS]				= {0.0};

float used_torque[DOF_JOINTS] = {0.0};

// task space controller variables
double maxJointChange = 0.03;
Eigen::VectorXd desiredTask[4];
Eigen::VectorXd taskPos[4];

Eigen::VectorXd jointState[4];
Eigen::VectorXd jointVel[4];
Eigen::VectorXd newJoints[4];

Eigen::MatrixXd jacobian[4];

KDL::Tree my_tree;
KDL::Chain fingerChains_[4];

boost::scoped_ptr<KDL::ChainFkSolverPos>    jnt_to_pose_solver_[4];
boost::scoped_ptr<KDL::ChainJntToJacSolver> jnt_to_jac_solver_[4];

KDL::JntArray  q_[4];            // Joint positions
KDL::Frame     x_[4];            // Tip pose
KDL::Jacobian  J_[4];            // Jacobian

Eigen::MatrixXd W;
double regularization;

ros::ServiceServer serviceWeightMatrix;
ros::ServiceServer serviceRegularization;

ros::ServiceServer triggerSimulationStart;

// ###############################################################################################################################
// VREP variables
// ###############################################################################################################################
#define MAX_EXT_API_CONNECTIONS 255
#define VREP_PORT 19997
char* ipVREP = "127.0.0.1";

int clientID;
simxInt jointHandles[DOF_JOINTS];
simxInt sensorHandles[4];
simxInt barHandle;

float dt = 0.010; // simulation timestep length
double maxTorque = 0.7;
bool simulationRunning = false;
int init = false;

bool trigger = false;

simxUChar **contactDataBuff;
simxInt contactDataLength;

simxUChar **contactDirectionsBuff;
simxInt contactDirectionsLength;

std::vector<float> fingercontact;
std::vector< std::vector<float> > contact;
std::vector<float> contactDirections;

int handlesCount;
int* handles;
int floatDataCount;
float* floatData;

// data conversion union
typedef union _data {
    float f;
    char  s[4];
  } charToFloat;

// if controller is enabeld, triggered with new incoming desired position, joint or task space
bool controlPD = false;
// if task space control should be use, otherwise joint control
bool controlTask = false;
// flag if current state should be saved to file
bool record = false;
int recordCounter = 0;
int recordLenght = 10; // timesteps to save
Eigen::MatrixXd recordJoints;
Eigen::MatrixXd recordSensors;

// Called when an external (string) message is received
void libCmdCallback(const std_msgs::String::ConstPtr& msg) {

	lib_cmd = msg->data.c_str();

	// turn on/off simulation
    if (lib_cmd.compare("save") == 0) {
        simulationRunning = !simulationRunning;
        if(simulationRunning) {
            for(int i = 0; i < DOF_JOINTS; i++)
                simxSetJointTargetPosition(clientID, jointHandles[i], DEGREES_TO_RADIANS(home_pose[i]), simx_opmode_oneshot);

            simxSynchronous(clientID,true);
            simxStartSimulation(clientID, simx_opmode_blocking);
            trigger = true;
            cout << "START simulation" << endl;
        }
        else{
            simxSetObjectIntParameter(clientID, barHandle, sim_shapeintparam_static, 1, simx_opmode_oneshot_wait);
            simxStopSimulation(clientID, simx_opmode_blocking);
            cout << "STOP simulation" << endl;
        }
    }
    else if (lib_cmd.compare("envelop") == 0) {
        simxSetObjectIntParameter(clientID, barHandle, sim_shapeintparam_static, 0, simx_opmode_oneshot_wait);
    }
    // manually trigger one simulation step
    else if(lib_cmd.compare("grasp_3") == 0) {
            trigger = true;
    }
    // record data and save to dummy file
    else if(lib_cmd.compare("record") == 0) {
        recordJoints = Eigen::MatrixXd::Zero(recordLenght, DOF_JOINTS);
        recordSensors = Eigen::MatrixXd::Zero(recordLenght, handStateMessage.sensorsPerFinger*4);
        recordCounter = 0;
        record = true;
    }

}

void SetTaskCallback(const sensor_msgs::JointState& msg) {

    for(int i=0; i < 4; i++)
        for(int k = 0; k < 7; k++)
            desiredTask[i][k] = msg.position[(7*i)+k];

    controlPD = true;
    trigger = true;
    controlTask = true;
}

void SetjointCallback(const sensor_msgs::JointState& msg){

    // cout << "msg IN: " << msg.header.seq << " " << msg.position[1] << endl;
	for(int i=0;i<DOF_JOINTS;i++) {
		desired_position[i] = msg.position[i];
    }


	controlPD = true;
    trigger = true;
    controlTask = false;
}

// starts/stops the simulation
bool triggerSimulationCB(allegro_hand_core_pd::triggerSimulation::Request &req, allegro_hand_core_pd::triggerSimulation::Response &res) {
    bool startSim = req.trigger;

    // simulationRunning = startSim;
    if(startSim) {
        // for(int i = 0; i < DOF_JOINTS; i++)
        //     simxSetJointTargetPosition(clientID, jointHandles[i], DEGREES_TO_RADIANS(home_pose[i]), simx_opmode_oneshot);

        simxSynchronous(clientID,true);
        simxStartSimulation(clientID, simx_opmode_blocking);
        trigger = true;
        simulationRunning = true;
        cout << "START simulation" << endl;
    }
    else {
        simxSetObjectIntParameter(clientID, barHandle, sim_shapeintparam_static, 1, simx_opmode_oneshot_wait);
        simxStopSimulation(clientID, simx_opmode_blocking);
        cout << "STOP simulation" << endl;
        trigger = false;
        simulationRunning = false;
    }
    usleep(50000);
    return true;
}

// sets the regularization
bool setRegularization(allegro_hand_core_cart::setRegularization::Request &req, allegro_hand_core_cart::setRegularization::Response &res) {

    regularization = req.lambda;
    return true;
}

// sets the weight matrix
bool setWeightMatrix(allegro_hand_core_cart::setWeightMatrix::Request &req, allegro_hand_core_cart::setWeightMatrix::Response &res) {
    for(int i = 0; i < 6; i++)
        W(i,i) = req.w[i];

    cout << "W: " << endl << W << endl;
    return true;
}

// service for getting the sensor directions
bool getSensorDirections(allegro_hand_core_pd::getSensorDirections::Request &req, allegro_hand_core_pd::getSensorDirections::Response &res) {

    res.vectors.resize(contactDirections.size());
    for(int i  = 0; i < contactDirections.size(); i++)
        res.vectors[i] = contactDirections.at(i);

    return true;
}

// initialize connection to VREP
bool initialize() {
    simxFinish(-1); // kill all old connections

    // open new connection and set to synchronous mode
    clientID = simxStart((simxChar*)ipVREP,VREP_PORT,true,true,2000,5);
	simxSynchronous(clientID,true);

    // set simulation timestep length
    simxSetFloatingParameter(clientID, sim_floatparam_simulation_time_step, dt, simx_opmode_oneshot_wait);

    // get joint handles for sending commands
    for(int i = 0; i < DOF_JOINTS; i++)
        simxGetObjectHandle(clientID,jointNames[i].c_str(),&jointHandles[i],simx_opmode_oneshot_wait);

    simxGetObjectHandle(clientID,"bar",&barHandle,simx_opmode_oneshot_wait);
    simxSetObjectIntParameter(clientID, barHandle, sim_shapeintparam_static, 1, simx_opmode_oneshot_wait);

    // set joints in streaming mode for reading
    simxGetObjectGroupData(clientID,
                            sim_object_joint_type,
                            15, // joint state data
                            &handlesCount, // handles count
                            &handles, // handles
                            NULL, // int data count
                            NULL, // int data
                            &floatDataCount, // float data count
                            &floatData, // float data
                            NULL, // string data count
                            NULL, // string data
                            simx_opmode_streaming); // opertion mode

    // set properties and start position of the hand
    for(int i = 0; i < DOF_JOINTS; i++) {
        simxSetJointTargetVelocity(clientID, jointHandles[i], 0.0, simx_opmode_oneshot_wait);
        simxSetJointForce(clientID,jointHandles[i], maxTorque, simx_opmode_oneshot_wait);
        simxSetObjectFloatParameter(clientID, jointHandles[i], sim_jointfloatparam_pid_p, 0.5*k_p[i], simx_opmode_oneshot_wait);
        simxSetObjectFloatParameter(clientID, jointHandles[i], sim_jointfloatparam_pid_d, 0.001*k_d[i], simx_opmode_oneshot_wait);
        simxSetObjectFloatParameter(clientID, jointHandles[i], sim_jointfloatparam_upper_limit, DEGREES_TO_RADIANS(360.0), simx_opmode_oneshot_wait);

        // simxSetJointTargetPosition(clientID, jointHandles[i], DEGREES_TO_RADIANS(home_pose[i]), simx_opmode_oneshot);
    }

    // sensor arrays
    //allocating the memory for receicing the data from V-rep application
    contactDirectionsBuff = new simxUChar* [400];
    for(int i = 0; i < 400; i++)
        contactDirectionsBuff[i] = new simxUChar[400];

    for(int i = 0; i < 400;i++)
        for(int j = 0; j < 400; j++)
            contactDirectionsBuff[i][j] = 0;

    contactDataBuff = new simxUChar* [400];
    for(int i = 0; i < 400; i++)
        contactDataBuff[i] = new simxUChar[400];

    // setting sensors in streaming mode
    for(int f = 0; f < 4; f++)
        simxGetStringSignal(clientID, sensorNames[f].c_str(), contactDataBuff, &contactDataLength,  simx_opmode_streaming);


    // start simulation
    // simxStartSimulation(clientID, simx_opmode_blocking);
    // simxSynchronousTrigger(clientID);

    simulationRunning = false;
    trigger = false;

    cout << "-----------------------------" << endl;
    cout << "KEYBOARD commands" << endl;
    cout << "\ts: start/stop simulation" << endl;
    cout << "-----------------------------" << endl;
    cout << "READY!" << endl;
    // cout << "START simulation" << endl;

    return true;
}


void initTaskController() {
    string robot_filename;
    ros::param::get("/robot_description", robot_filename);

    if (!kdl_parser::treeFromString(robot_filename, my_tree))
       ROS_ERROR("Failed to construct kdl tree");
    else
       cout << "\n#################################\nURDF model loaded\n" << endl;


    my_tree.getChain("base_link", "link_3.0_tip", fingerChains_[0]); // index
    my_tree.getChain("base_link", "link_7.0_tip", fingerChains_[1]); // middle
    my_tree.getChain("base_link", "link_11.0_tip", fingerChains_[2]); // little
    my_tree.getChain("base_link", "link_15.0_tip", fingerChains_[3]); // thumb


    for(int i = 0; i < 4; i++) {
        jacobian[i] = Eigen::MatrixXd::Zero(6,4);
        taskPos[i] = Eigen::VectorXd::Zero(7);

        jointState[i] = Eigen::VectorXd::Zero(4);
        jointVel[i] = Eigen::VectorXd::Zero(4);

        desiredTask[i] = Eigen::VectorXd::Zero(7);
        newJoints[i] = Eigen::VectorXd::Zero(4);

        jnt_to_pose_solver_[i].reset(new KDL::ChainFkSolverPos_recursive(fingerChains_[i]));
        jnt_to_jac_solver_[i].reset(new KDL::ChainJntToJacSolver(fingerChains_[i]));

        q_[i].resize(fingerChains_[i].getNrOfJoints());
        J_[i].resize(fingerChains_[i].getNrOfJoints());
    }

    W = 1.0 * Eigen::MatrixXd::Identity(6,6);
    W.block(3,3,3,3) = 0.2 * Eigen::MatrixXd::Identity(3,3); // scale orientation
    regularization = 1e-3;
    cout << "W: " << endl << W << endl;
}


// calculate new joint positions given task positioni
void calculateNewJoints() {
    for(int i = 0; i < 4; i++) {
        for (unsigned int j = 0; j < 6; j++) {
            Eigen::VectorXd temp(4);
			for (unsigned int k = 0; k < 4; k++) {
                temp[k] = J_[i](j,k);
                jacobian[i].row(j) = temp;
            }
        }

        // get difference
        Eigen::VectorXd curPos(3);
        curPos << taskPos[i][0], taskPos[i][1], taskPos[i][2];
        Eigen::VectorXd desPos(3);
        desPos << desiredTask[i][0], desiredTask[i][1], desiredTask[i][2];

        Eigen::VectorXd posVel = desPos - curPos;

        Eigen::Quaterniond curOr(taskPos[i][6],taskPos[i][3],taskPos[i][4],taskPos[i][5]);
        curOr.normalize();
        Eigen::Quaterniond desOr(desiredTask[i][6],desiredTask[i][3],desiredTask[i][4],desiredTask[i][5]);
        desOr.normalize();

        Eigen::Quaterniond angVel = desOr * curOr.conjugate();
        angVel.vec() = angVel.vec() *2;


        Eigen::VectorXd cartVel(6);
        cartVel << posVel[0], posVel[1], posVel[2], angVel.x(), angVel.y(), angVel.z();

        Eigen::MatrixXd ridentity = regularization*Eigen::MatrixXd::Identity(4, 4);
        Eigen::VectorXd jointVel = (jacobian[i].transpose() * W * jacobian[i] + ridentity).colPivHouseholderQr().solve(jacobian[i].transpose() * W * cartVel);

        jointVel =  jointVel.cwiseMin(maxJointChange);
        jointVel =  jointVel.cwiseMax(-maxJointChange);


        newJoints[i] = jointState[i] + jointVel;
    }

    for(int f = 0; f < 4; f++)
        for(int k = 0; k < 4; k++)
            desired_position[(4*f)+k] = newJoints[f][k];
}


// update controller
void updateController() {
    // save last iteration info
	for (int i=0; i<DOF_JOINTS; i++) {
		previous_position[i] = current_position[i];
		previous_position_filtered[i] = current_position_filtered[i];
		previous_velocity[i] = current_velocity[i];
	}

    // if(floatData)
    // cout << "data OLD: " << " " << floatData[1*2] << " " <<  current_position[1] << " " << current_position_filtered[1] << endl;

    // read new states
    int ret = simxGetObjectGroupData(clientID,
                            sim_object_joint_type,
                            15, // joint state data
                            &handlesCount, // handles count
                            &handles, // handles
                            NULL, // int data count
                            NULL, // int data
                            &floatDataCount, // float data count
                            &floatData, // float data
                            NULL, // string data count
                            NULL, // string data
                            simx_opmode_buffer); // opertion mode

    if(ret == simx_return_ok) {
        for (int i=0; i<DOF_JOINTS; i++) {
            current_position[i] = floatData[i*2];
            used_torque[i] = floatData[(i*2)+1];

            current_position_filtered[i] = (0.6*current_position_filtered[i]) + (0.198*previous_position[i]) + (0.198*current_position[i]);
            current_velocity[i] = (current_position_filtered[i] - previous_position_filtered[i]) / dt;
            current_velocity_filtered[i] = (0.6*current_velocity_filtered[i]) + (0.198*previous_velocity[i]) + (0.198*current_velocity[i]);
            current_velocity[i] = (current_position[i] - previous_position[i]) / dt;
        }
    }
    // cout << "data NEW: " << " " << floatData[1*2] << " " <<  current_position[1] << " " << current_position_filtered[1] << endl;

    contact.resize(0);
    for(int f = 0; f < 4; f++) {
        fingercontact.resize(0);

        for(int i = 0; i < 400;i++)
            for(int j = 0; j < 400; j++)
                contactDataBuff[i][j] = 0;

        int ret = simxGetStringSignal(clientID, sensorNames[f].c_str(), contactDataBuff, &contactDataLength,  simx_opmode_buffer);

        if(ret == simx_return_ok) {
            for(int i = 0; i < contactDataLength/4; i++) {
                charToFloat contactdata;
                for(int j = 0; j < 4; j++)
                    contactdata.s[j]= contactDataBuff[0][i*4 + j];
                fingercontact.push_back(contactdata.f);
            }
            contact.push_back(fingercontact);
        }
    }

    // get FK and jacobian
    for(int f = 0; f < 4; f++) {
        for(int k = 0; k < 4; k++) {
            q_[f](k) = current_position[(4*f)+k];
            jointState[f][k] = q_[f](k);
        }
        jnt_to_jac_solver_[f]->JntToJac(q_[f], J_[f]);
        jnt_to_pose_solver_[f]->JntToCart(q_[f],x_[f]);

        taskPos[f][0] = x_[f].p.x();
        taskPos[f][1] = x_[f].p.y();
        taskPos[f][2] = x_[f].p.z();
        double qx, qy, qz, qw;
		x_[f].M.GetQuaternion(qx, qy, qz, qw);
        taskPos[f][3] = qx;
        taskPos[f][4] = qy;
        taskPos[f][5] = qz;
        taskPos[f][6] = qw;
    }

    if(controlTask)
        calculateNewJoints();

    // set target position if enabled
    if(controlPD) {
        for(int i = 0; i < DOF_JOINTS; i++) {
            simxSetJointTargetPosition(clientID, jointHandles[i], desired_position[i], simx_opmode_oneshot);
        }
        controlPD = false;
    }
}

// publish all data
void publishData() {
    // current position, velocity and effort (torque) published
    handStateMessage.header.stamp = ros::Time::now();
    handStateMessage.jointState.header = handStateMessage.header;
    for (int i=0; i<DOF_JOINTS; i++) {
        handStateMessage.jointState.position[i] = current_position[i]; // current_position_filtered[i];
        handStateMessage.jointState.velocity[i] = current_velocity[i]; // current_velocity_filtered[i];
        handStateMessage.jointState.effort[i] = used_torque[i];

        if(record)
            recordJoints(recordCounter,i) = current_position[i];
    }
    // cout << "msg OUT: " << handStateMessage.header.seq << " " << handStateMessage.jointState.position[1] << " " << current_position_filtered[1] << endl;

    handStateMessage.taskState.header = handStateMessage.header;
    for(int f = 0; f < 4; f++)
        for(int k = 0; k < 7; k++)
            handStateMessage.taskState.position[(7*f)+k] = taskPos[f][k];

    for(int f = 0; f < contact.size(); f++) {
        for(int i = 0; i < contact.at(f).size(); i++) {
            handStateMessage.sensor_data[(f*handStateMessage.sensorsPerFinger)+i] = contact.at(f).at(i);

            if(record)
                recordSensors(recordCounter,(f*handStateMessage.sensorsPerFinger)+i) = contact.at(f).at(i);
        }
    }
    state_publisher.publish(handStateMessage);

    if(record) {
        recordCounter++;
        if(recordCounter == recordLenght) {
            record = false;

            ofstream myfile;
            myfile.open("/home/tanneberg/allegro_ws/allegro_hand_ros/demonstration.txt");
            myfile << recordJoints.colwise().mean() << endl << endl;
            myfile << recordSensors.colwise().mean() << endl << endl;
            myfile.close();

            // cout << recordJoints.colwise().mean() << endl << endl;
            // cout << recordSensors.colwise().mean() << endl << endl;
        }
    }
}


// main controller loop
void mainLoop() {
    // get tactile point directions once in the beginning
    if(!init) {
        int ret = simxGetStringSignal(clientID, "tactilePointDirections", contactDirectionsBuff, &contactDirectionsLength,  simx_opmode_oneshot_wait);
        if(ret == simx_return_ok) {
            int counter = 0;
            int lcounter = 0;
            for(int i = 0; i < contactDirectionsLength/4; i++) {
                charToFloat contactdir;
                for(int j = 0; j < 4; j++)
                    contactdir.s[j]= contactDirectionsBuff[0][i*4 + j];
                contactDirections.push_back(contactdir.f);
                if(i % 3 == 0) {
                    counter++;
                }
            }
            lcounter++;
            if(lcounter == 3) {
                lcounter = 0;
            }

        handStateMessage.sensorsPerFinger = counter;
        handStateMessage.sensor_data.resize(4*counter);
        init = true;

        }
    }

    // start mainLoop
    if(simulationRunning && trigger) {
        updateController();
        simxSynchronousTrigger(clientID);
        publishData();
        trigger = false;
    }
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "VREP_allegro_node");
	ros::Time::init();

	ros::NodeHandle nh;

    // init ROS stuff
    state_publisher = nh.advertise<allegro_hand_core_pd::handState>(STATE_TOPIC, 1);
    joint_cmd_sub = nh.subscribe(JOINT_CMD_TOPIC, 1, SetjointCallback, ros::TransportHints().tcpNoDelay());
    task_cmd_sub = nh.subscribe(TASK_CMD_TOPIC, 1, SetTaskCallback, ros::TransportHints().tcpNoDelay());
    lib_cmd_sub = nh.subscribe(LIB_CMD_TOPIC, 1, libCmdCallback, ros::TransportHints().tcpNoDelay());

    serviceSensorDirections = nh.advertiseService("get_sensor_directions", getSensorDirections);
    triggerSimulationStart = nh.advertiseService("triggerSimulation", triggerSimulationCB);

    // Create arrays 16 long for each of the four joint state components
    handStateMessage.jointState.position.resize(DOF_JOINTS);
    handStateMessage.jointState.velocity.resize(DOF_JOINTS);
    handStateMessage.jointState.effort.resize(DOF_JOINTS);
    handStateMessage.jointState.name.resize(DOF_JOINTS);

    handStateMessage.taskState.position.resize(4*7);

    for (int i=0; i<DOF_JOINTS; i++)
        handStateMessage.jointState.name[i] = jointNames[i];

    initTaskController();
    serviceRegularization = nh.advertiseService("set_regularization", setRegularization);
    serviceWeightMatrix = nh.advertiseService("set_weight_matrix", setWeightMatrix);

    if(initialize()) {
    	while (ros::ok()) {
            mainLoop();
    		ros::spinOnce();
    	}
    }

    // clean up
    simxStopSimulation(clientID, simx_opmode_blocking);
    simxFinish(clientID);
	nh.shutdown();

	return 0;
}
