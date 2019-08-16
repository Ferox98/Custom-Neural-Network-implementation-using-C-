#pragma once

#include <Eigen/Dense>

#include <vector>
#include <math.h>
#include <tuple>
#include <random>
#include <algorithm>

using namespace std;
using namespace Eigen;

typedef tuple<vector<vector<double>>, double> Instance;
typedef vector<Instance> Data; // the entire dataset

class Network {

private:
	int num_layers;
	vector<int> num_neurons;
	double eta = 0.01; // default learning rate, 
	// matrices for the weights, activations and biases
	vector<MatrixXd> weight_matrices;
	vector<VectorXd> bias_vectors;

public:
	Network(vector<int>& layers);
	
	inline double sigmoid(double x) {
		// compute the sigmoid function
		return 1.0 / (1.0 + exp(-x));
	}

	VectorXd& sigmoidPrime(VectorXd& vec);
	
	VectorXd& costDerivative(VectorXd output, double y);

	void SGD(int epochs, int mini_batch_size, Data& training_data, double eta);
	
	void updateNetwork(Data& mini_batches);

	tuple<vector<MatrixXd>, vector<VectorXd>> backpropagate(vector<vector<double>>& x, double y);
};
