#include "ros/ros.h"
#include "ros/service.h"
#include "ros/service_server.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/String.h"

#include "allegro_hand_core_pd/handState.h"
#include "allegro_hand_core_pd/triggerSimulation.h"

#include <boost/scoped_ptr.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include "mat.h"
#include "matrix.h"

#include <Eigen/Dense>

#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#define DOF_JOINTS 16
using namespace std;

const char *inputFile = "/dev/shm/controller.mat";
const char *outputFile = "/dev/shm/rolloutdata.mat";
const char *outputFileTemp = "/dev/shm/rolloutdataTEMP.mat";

// Topics
#define STATE_TOPIC "/allegroHand/state"

#define JOINT_CMD_TOPIC "/allegroHand/joint_cmd"
#define TASK_CMD_TOPIC "/allegroHand/task_cmd"
#define LIB_CMD_TOPIC "/allegroHand/lib_cmd"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

double home_pose[DOF_JOINTS] =
{
	0.0,  -10.0,   45.0,   45.0,  // Default (HOME) position (degrees)
	0.0,  -10.0,   45.0,   45.0,	// This position is loaded and set upon system start
	5.0,   -5.0,   50.0,   45.0,	// if no 'initial_position.yaml' parameter is loaded.
	105.0,  -10.0,   5.0,   10.0
};

ros::Subscriber state_sub;

ros::Publisher joint_cmd_pub;
ros::Publisher task_cmd_pub;
ros::Publisher lib_cmd_pub;

// messages
allegro_hand_core_pd::handState handStateMessage;
string lib_cmd;
allegro_hand_core_pd::triggerSimulation startStopSim;

sensor_msgs::JointState msgJoint;

// parameters
int numberOfRollouts; // set automatically
int resendSteps = 10;

// helper variables
struct tm* initTM;
struct stat initAtt;
time_t initTime;

int rolloutsCounter;
int timestepCounter;
int resendCounter;
bool trigger;
bool newRollout;
bool enabled;
int startPosCounter;


KDL::Tree my_tree;
KDL::Chain fingerChains_[4];

boost::scoped_ptr<KDL::ChainFkSolverPos>    jnt_to_pose_solver_[4];

KDL::JntArray  q_[4];            // Joint positions
KDL::Frame     x_[4];            // Tip pose
Eigen::VectorXd taskPos[4];
float taskDist;
bool useTaskConstraint = false;
double maxAction = 0.1;

// ##############################################################################
// input variables
Eigen::VectorXd b;
double beta;
Eigen::MatrixXd cholA;
Eigen::VectorXd mask_state_C;
Eigen::VectorXd mask_state_J;
Eigen::VectorXd mask_state_T;
Eigen::MatrixXd polMean;
Eigen::MatrixXd omega;
Eigen::VectorXd rolloutLengths;
int startPosTimesteps;
Eigen::VectorXd startPosition;
Eigen::MatrixXd startPositionsMat;

// ##############################################################################
// output variables
// state matrices
Eigen::MatrixXd sjMat;
Eigen::MatrixXd stMat;
// action matrix
Eigen::MatrixXd aMat;
// sensor MatrixXd
Eigen::MatrixXd cMat;
// used parameter matrix
Eigen::MatrixXd paMat;
Eigen::MatrixXd oldParas;

// ##############################################################################
// random number functions
//Box muller method
Eigen::VectorXd rand_normal(double mean, double stddev, int numbers) {
	Eigen::VectorXd randVector = Eigen::VectorXd::Zero(numbers);

	for(int i = 0; i < numbers; i++) {
	    static double n2 = 0.0;
	    static int n2_cached = 0;
	    if (!n2_cached) {
	        double x, y, r;
	        do {
	            x = 2.0*rand()/RAND_MAX - 1;
	            y = 2.0*rand()/RAND_MAX - 1;

	            r = x*x + y*y;
	        }
	        while (r == 0.0 || r > 1.0);
	        {
	            double d = sqrt(-2.0*log(r)/r);
	            double n1 = x*d;
	            n2 = y*d;
	            double result = n1*stddev + mean;
	            n2_cached = 1;
	            // return result;
				randVector[i] = result;
	        }
	    }
	    else {
	        n2_cached = 0;
	        // return n2*stddev + mean;
			randVector[i] = n2*stddev + mean;
	    }
	}

	return randVector;
}

// sample from multivariate gaussian
Eigen::MatrixXd rand_multi(Eigen::MatrixXd mean, Eigen::MatrixXd cholCov) {
	Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mean.rows(), mean.cols());

	for(int r = 0; r < mean.rows(); r++) {
		Eigen::VectorXd normal_randoms = rand_normal(0, 1, mean.cols());
		Eigen::VectorXd oneActionPara = mean.row(r).transpose() + (cholCov * normal_randoms);
		result.row(r) = oneActionPara.transpose();
	}

	return result;
}

// ##############################################################################
// .mat files functions
void readMatFile(const char *file) {
	// open MAT-file
	printf("READING: %s .. ", file);
	MATFile *pmat = matOpen(file, "r");
	if (pmat == NULL) {
		// cout << "TESE" << endl;
		return;
	}

	// extract the specified variables
	// ###########################################################################
	mxArray *arr = matGetVariable(pmat, "b");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);

		b = Eigen::VectorXd::Zero(dims[0]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[0]; i++)
				b(i) = pr[i];
		}

		// cout << "b :\n" <<b << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "beta");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			beta = pr[0];
		}
		// cout << "beta: " << beta << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "cholA");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);
		// cout << "DIMENSIONS of cholA: " << dims[0] << " " << dims[1] << endl;

		cholA = Eigen::MatrixXd::Zero(dims[0],dims[1]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[1]; i++)
				for(int j = 0; j < dims[0]; j++)
					cholA(j,i) = pr[(i*dims[0])+j];
		}

		// cout << "cholA :\n" << cholA << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "mask_state_C");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);

		mask_state_C = Eigen::VectorXd::Zero(dims[0]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[0]; i++)
				mask_state_C(i) = pr[i];
		}

		// cout << "mask_state_C :\n" << mask_state_C << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "mask_state_J");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);

		mask_state_J = Eigen::VectorXd::Zero(dims[0]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[0]; i++)
				mask_state_J(i) = pr[i];
		}

		// cout << "mask_state_J :\n" << mask_state_J << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "mask_state_T");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);

		mask_state_T = Eigen::VectorXd::Zero(dims[0]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[0]; i++)
				mask_state_T(i) = pr[i];
		}

		// cout << "mask_state_T :\n" << mask_state_T << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "mean");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);
		// cout << "DIMENSIONS of mean: " << dims[0] << " " << dims[1] << endl;

		polMean = Eigen::MatrixXd::Zero(dims[0],dims[1]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[1]; i++)
				for(int j = 0; j < dims[0]; j++)
					polMean(j,i) = pr[(i*dims[0])+j];
		}

		// cout << "mean :\n" << polMean << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "omega");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);
		// cout << "DIMENSIONS of omega: " << dims[0] << " " << dims[1] << endl;

		omega = Eigen::MatrixXd::Zero(dims[0],dims[1]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[1]; i++)
				for(int j = 0; j < dims[0]; j++)
					omega(j,i) = pr[(i*dims[0])+j];
		}

		// cout << "omega :\n" << omega << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "rolloutLenghts");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);

		rolloutLengths = Eigen::VectorXd::Zero(dims[0]);
		numberOfRollouts = dims[0];

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[0]; i++)
				rolloutLengths(i) = pr[i];
		}

		// cout << "rolloutLengths :\n" <<rolloutLengths << endl;
	}
	mxDestroyArray(arr);

	// ###########################################################################
	arr = matGetVariable(pmat, "startPositionTimesteps");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			startPosTimesteps = pr[0];
		}
		// cout << "startPositionTimesteps: " << startPosTimesteps << endl;
	}
	mxDestroyArray(arr);


	// ###########################################################################
	arr = matGetVariable(pmat, "startPositions");
	if (arr != NULL && mxIsDouble(arr) && !mxIsEmpty(arr)) {

		const mwSize *dims = mxGetDimensions(arr);
		// cout << "DIMENSIONS of startPositions: " << dims[0] << " " << dims[1] << endl;

		startPositionsMat = Eigen::MatrixXd::Zero(dims[0],dims[1]);

		double *pr = mxGetPr(arr);
		if (pr != NULL) {
			for(int i = 0; i < dims[1]; i++)
				for(int j = 0; j < dims[0]; j++)
					startPositionsMat(j,i) = pr[(i*dims[0])+j];
		}

		// cout << "startPositionsMat :\n" << startPositionsMat << endl;
	}
	mxDestroyArray(arr);


	// cleanup
	matClose(pmat);

	printf("Finished.\n");
}

void updateMatFile(const char *file, int index) {
	printf("UPDATING: %s | Rollout: %d/%d .. ", file, index+1, numberOfRollouts);
	MATFile *pmat = matOpen(file, "u");
	if (pmat == NULL) return;

	// state joint matrix
	mxArray *pa1 = mxCreateDoubleMatrix(sjMat.rows(),sjMat.cols(),mxREAL);
	double *pr1 = mxGetPr(pa1);
	memcpy((void *)pr1, sjMat.data(), (sjMat.rows() * sjMat.cols()) * sizeof(sjMat.data()));

	// state task matrix
	mxArray *pa4 = mxCreateDoubleMatrix(stMat.rows(),stMat.cols(),mxREAL);
	double *pr4 = mxGetPr(pa4);
	memcpy((void *)pr4, stMat.data(), (stMat.rows() * stMat.cols()) * sizeof(stMat.data()));

	// action matrix
	mxArray *pa2 = mxCreateDoubleMatrix(aMat.rows(),aMat.cols(),mxREAL);
	double *pr2 = mxGetPr(pa2);
	memcpy((void *)pr2, aMat.data(), (aMat.rows() * aMat.cols()) * sizeof(aMat.data()));

	// sensor matrix
	mxArray *pa3 = mxCreateDoubleMatrix(cMat.rows(),cMat.cols(),mxREAL);
	double *pr3 = mxGetPr(pa3);
	memcpy((void *)pr3, cMat.data(), (cMat.rows() * cMat.cols()) * sizeof(cMat.data()));

	// parameter matrix
	mxArray *pa5 = mxCreateDoubleMatrix(paMat.rows(),paMat.cols(),mxREAL);
	double *pr5 = mxGetPr(pa5);
	memcpy((void *)pr5, paMat.data(), (paMat.rows() * paMat.cols()) * sizeof(paMat.data()));

	// get cell array
	mxArray *cArray = matGetVariable(pmat, "rolloutsArray_J_T_A_S_P");
	const mwSize *dims = mxGetDimensions(cArray);

	// add entry, state and and action matrix
	mxSetCell(cArray, index, mxDuplicateArray(pa1)); // joint states
	mxSetCell(cArray, (1*dims[0])+index, mxDuplicateArray(pa4)); // task states
	mxSetCell(cArray, (2*dims[0])+index, mxDuplicateArray(pa2)); // actions
	mxSetCell(cArray, (3*dims[0])+index, mxDuplicateArray(pa3)); // sensors
	mxSetCell(cArray, (4*dims[0])+index, mxDuplicateArray(pa5)); // paramters
	matPutVariable(pmat, "rolloutsArray_J_T_A_S_P", cArray);

	// clean up
	matClose(pmat);
	mxDestroyArray(pa1);
	mxDestroyArray(pa2);
	mxDestroyArray(pa3);
	mxDestroyArray(pa4);
	mxDestroyArray(pa5);
	mxDestroyArray(cArray);

	printf("Finished.\n");
}

void writeMatFile(const char *file, int numRollouts) {

    printf("CREATING: %s .. ", file);
	MATFile *pmat = matOpen(file, "w");
	if (pmat == NULL) return;

	mwSize ndim = 2;
	mwSize *dims = new mwSize[ndim];
	dims[0] = numRollouts;
	dims[1] = 5;
	mxArray *cArray = mxCreateCellArray(ndim, dims);

	matPutVariable(pmat, "rolloutsArray_J_T_A_S_P", cArray);

	matClose(pmat);
	mxDestroyArray(cArray);

	printf("Finished.\n");
}

bool checkFileUpdate() {
	usleep(150000);
	struct tm* clock;
	struct stat att;
	stat(inputFile, &att);
	clock = gmtime(&(att.st_mtime));
	time_t newTime = mktime(clock);

	if(difftime(newTime, initTime) > 0) {
		initTime = newTime;
		return true;
	}
	return false;
}

// ##############################################################################
void setOutputAction() {
	msgJoint.header.stamp = ros::Time::now();

	for(int i = 0; i < 16; i++)
		msgJoint.position[i] = startPosition(i);

	for(int i = 0; i < mask_state_J.size(); i++) {
		if(!useTaskConstraint || taskDist < 0.12)
			msgJoint.position[mask_state_J[i]] = sjMat(timestepCounter,mask_state_J[i]) + aMat(timestepCounter,i);
		else
			msgJoint.position[mask_state_J[i]] = sjMat(timestepCounter,mask_state_J[i]);
	}

}


// get features
Eigen::VectorXd calcFeatures(Eigen::VectorXd states) {
	Eigen::VectorXd z = sqrt(2.0*(1.0/b.size())) * ((omega * states) + b).array().cos();

	return z;
}

// policy
void calculateNewAction() {

	Eigen::VectorXd states = Eigen::VectorXd::Zero(mask_state_J.size() + mask_state_C.size());
	for(int i = 0; i < mask_state_J.size(); i++)
		states[i] = sjMat(timestepCounter,mask_state_J[i]);
	for(int i = mask_state_J.size(); i < mask_state_C.size(); i++)
		states[i] = cMat(timestepCounter,mask_state_C[i]);

	Eigen::VectorXd features = calcFeatures(states);
	Eigen::MatrixXd parameters;
	Eigen::VectorXd newActions;

	if(timestepCounter == 0) {
		parameters = rand_multi(polMean, cholA);
		newActions = parameters * features;
	}
	else {
		if(beta > 0) {
			parameters = rand_multi(polMean, ((2.0/beta)-1.0)*cholA);
			newActions = (((1.0 - beta)*oldParas) + (beta*parameters)) * features;
		}
		else {
			parameters = paMat.row(timestepCounter-1);
			newActions = parameters * features;
		}
	}

	oldParas = Eigen::MatrixXd(parameters);
	paMat.row(timestepCounter) = Eigen::VectorXd(parameters);

	for(int i = 0; i < aMat.cols(); i++) {
		aMat(timestepCounter, i) = min(max(newActions[i], -1*maxAction), maxAction);
	}

	// ################################################################################
	// TASK SPACE CONSTRAINT
	// get FK and jacobian
    for(int f = 0; f < 4; f++)
        for(int k = 0; k < 4; k++)
            q_[f](k) = sjMat(timestepCounter, (4*f)+k);

	int f,k;
	for(int i = 0; i < mask_state_J.size(); i++) {
		if(mask_state_J[i] <= 3) { // 0 1 2 3
			f = 0;
			k = mask_state_J[i];
		}
		else if(mask_state_J[i] <= 7) { // 4 5 6 7
			f = 1;
			k = mask_state_J[i] - 4;
		}
		else if(mask_state_J[i] <= 11) { // 8 9 10 11
			f = 2;
			k = mask_state_J[i] - 8;
		}
		else { // 12 13 14 15
			f = 3;
			k = mask_state_J[i] - 12;
		}
		q_[f](k) = q_[f](k) + aMat(timestepCounter,i);
	}
	for(int f = 0; f < 4; f++) {
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

	float indx = taskPos[0][0];
	float indz = taskPos[0][2];
	float midx = taskPos[1][0];
	float midz = taskPos[1][2];
	float thux = taskPos[3][0];
	float thuz = taskPos[3][2];
	taskDist = sqrt(pow(indx-thux,2) + pow(indz-thuz,2));


	// ################################################################################

	setOutputAction();
}

// ##############################################################################
// callbacks
void stateCallback(const allegro_hand_core_pd::handState& msg) {

	if(enabled) {
		if(startPosCounter < startPosTimesteps) {
			msgJoint.header.stamp = ros::Time::now();

			for(int i = 0; i < 16; i++)
				msgJoint.position[i] = startPosition(i);

			startPosCounter++;

			// to enable dynamic simulation of object
			if(startPosCounter == startPosTimesteps) {
				startPosCounter++;
				lib_cmd = "envelop";
				lib_cmd_pub.publish(lib_cmd);
			}
		}
		else {
			resendCounter++;

			if(resendCounter == resendSteps) {
				// save data into matrices
				for(int i = 0; i < sjMat.cols(); i++)
					sjMat(timestepCounter, i) = msg.jointState.position[i];

				for(int i = 0; i < stMat.cols(); i++)
					stMat(timestepCounter, i) = msg.taskState.position[i];

				for(int i = 0; i < cMat.cols(); i++)
					cMat(timestepCounter, i) = msg.sensor_data[i];

				calculateNewAction();

				resendCounter = 0;
				timestepCounter++;
				if(timestepCounter == rolloutLengths[rolloutsCounter]) {

					updateMatFile(outputFileTemp, rolloutsCounter);

					rolloutsCounter++;
					startStopSim.request.trigger = false;
					ros::service::call("triggerSimulation", startStopSim);
					newRollout = true;

					if(rolloutsCounter == numberOfRollouts) {
						ifstream  src(outputFileTemp, ios::binary);
					    ofstream  dst(outputFile, ios::binary);

					    dst << src.rdbuf();
					}
				}
			}
		}
		trigger = true;
	}
}


// ##############################################################################
int main(int argc, char** argv) {

	ros::init(argc, argv, "learned_controller_node");
	ros::NodeHandle nh;

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
        jnt_to_pose_solver_[i].reset(new KDL::ChainFkSolverPos_recursive(fingerChains_[i]));
        q_[i].resize(fingerChains_[i].getNrOfJoints());
		taskPos[i] = Eigen::VectorXd::Zero(7);
	}

	msgJoint.position.resize(16);
	for(int i = 0; i < 16; i++)
		msgJoint.position[i] = 0.0;

	state_sub = nh.subscribe(STATE_TOPIC, 1, stateCallback, ros::TransportHints().tcpNoDelay());
	lib_cmd_pub = nh.advertise<std_msgs::String>(LIB_CMD_TOPIC, 1);
	joint_cmd_pub = nh.advertise<sensor_msgs::JointState>(JOINT_CMD_TOPIC, 1);

	rolloutsCounter = 0;
	timestepCounter = 0;
	resendCounter = 0;
	startPosCounter = 0;

	readMatFile(inputFile);
	writeMatFile(outputFileTemp, numberOfRollouts);

	/// get init timestamp
	stat(inputFile, &initAtt);
	initTM = gmtime(&(initAtt.st_mtime));
	initTime = mktime(initTM);

	enabled = true;
	newRollout = true;
	lib_cmd = "grasp_3";
	while (ros::ok()) {
		if(rolloutsCounter == numberOfRollouts) {
			if(checkFileUpdate()) {
				cout << "NEW parameters" << endl;

				rolloutsCounter = 0;
				timestepCounter = 0;
				resendCounter = 0;
				startPosCounter = 0;
				newRollout = true;

				readMatFile(inputFile);
				writeMatFile(outputFileTemp, numberOfRollouts);
				enabled = true;
			}
			else {
				trigger = false;
				enabled = false;
			}
		}

		if(newRollout && enabled) {
			startStopSim.request.trigger = true;
			ros::service::call("triggerSimulation", startStopSim);

			// Initialize rollout matrices
			sjMat = Eigen::MatrixXd::Zero(rolloutLengths[rolloutsCounter], 16); // 16 joint
			stMat = Eigen::MatrixXd::Zero(rolloutLengths[rolloutsCounter], 28); // 4*7 task positions
			aMat = Eigen::MatrixXd::Zero(rolloutLengths[rolloutsCounter], mask_state_J.size());
			cMat = Eigen::MatrixXd::Zero(rolloutLengths[rolloutsCounter], 72); // 18*4 sensors
			paMat = Eigen::MatrixXd::Zero(rolloutLengths[rolloutsCounter], polMean.rows()* polMean.cols()); // sampled parameters
			newRollout = false;

			startPosition = startPositionsMat.row(rolloutsCounter);

			timestepCounter = 0;
			startPosCounter = 0;

		}

		if(trigger && enabled) {
			joint_cmd_pub.publish(msgJoint);
			trigger = false;
		}


		if(enabled)
			usleep(10);

		ros::spinOnce();
	}
	startStopSim.request.trigger = false;
	ros::service::call("triggerSimulation", startStopSim);

	nh.shutdown();

	return 0;
}
