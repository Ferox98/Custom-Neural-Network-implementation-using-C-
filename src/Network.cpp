#include "Network.h"


Network::Network(vector<int>& layers) {
	num_layers = layers.size();
	num_neurons = vector<int>(num_layers);
	// initialize the weight, activation and bias matrices
	for (int i = 0; i < layers.size(); i++) {
		if (i != layers.size() - 1)
			weight_matrices[i] = MatrixXd::Random(layers[i], layers[i + 1]);
		if (i != 0) {
			// input layer has no activation or bias vector
			bias_vectors[i] = VectorXd::Random(layers[i]); // a column vector with specified dimensions and 0-1 randomly initialized values
		}
		num_neurons[i] = layers[i];
	}
}

VectorXd& Network::sigmoidPrime(VectorXd& vec) {
	VectorXd activation(vec.rows());
	for (int i = 0; i < vec.rows(); i++)
		activation(i) = sigmoid(vec(i)) * (1 - sigmoid(vec(i)));
	return activation;
}

VectorXd& Network::costDerivative(VectorXd output, double y) {
	// return a vector of partial derivatives partial Cx / partial a(k) 
	VectorXd vec(output.rows());
	for (int i = 0; i < output.rows(); i++) {
		vec(i) = (output(i) - 1.0) ? (i + 1) == y : (output(i));
	}
	return vec;
}

void Network::SGD(int epochs, int mini_batch_size, Data& training_data, double eta) {
	// This function performs stochastic gradient descent on the network using specified mini-batch size
	for (int i = 0; i < epochs; i++) {
		auto rng = default_random_engine{};
		shuffle(begin(training_data), end(training_data), rng);
		// select mini-batch
		vector<Data> mini_batches;
		for (int j = 0; j < training_data.size(); j += mini_batch_size) {
			Data mini_batch;
			for (int k = j; k < j + mini_batch_size; k++)
				mini_batch.push_back(training_data[k]);
			mini_batches.push_back(mini_batch);
		}
		// update the network for each mini-batch
		for (auto& mini_batch : mini_batches) {
			updateNetwork(mini_batch);
		}
	}
}

void Network::updateNetwork(Data& mini_batch) {
	// retrieves gradients for the weights and biases and updates the network with them
	int len = mini_batch.size();

	for (auto& _tuple : mini_batch) {
		vector<vector<double>>& x = get<0>(_tuple);
		double y = get<1>(_tuple);
		tuple<vector<MatrixXd>, vector<VectorXd>> matrices = backpropagate(x, y);
		vector<MatrixXd>& weight_gradients = get<0>(matrices);
		vector<VectorXd>& bias_gradients = get<1>(matrices);

		for (int i = 0; i < weight_gradients.size(); i++)
			weight_matrices[i] -= (eta / (double)len) * weight_gradients[i];

		for (int i = 0; i < bias_gradients.size(); i++)
			bias_vectors[i] -= (eta / (double)len) * bias_gradients[i];
	}
}

tuple<vector<MatrixXd>, vector<VectorXd>> Network::backpropagate(vector<vector<double>>& x, double y) {
	// feed-forward
	VectorXd activation(x.size() * x[0].size());
	vector<VectorXd> activations = vector<VectorXd>(num_layers);
	int counter = 0;
	for (int i = 0; i < x.size(); i++)
		for (int j = 0; j < x[i].size(); j++)
			activation(counter) = x[i][j];

	vector<VectorXd> z_vectors;
	for (int i = 0; i < num_layers; i++) {
		activations.push_back(activation);
		VectorXd z = activation * weight_matrices[i] + bias_vectors[i];
		activation = VectorXd(num_neurons[i + 1]);
		for (int j = 0; j < z.rows(); j++) {
			double sigmoid_x = sigmoid(z[j]);
			activation[j] = sigmoid_x;
		}
		z_vectors.push_back(z);
	}

	// perform backward pass
	// initialize gradient matrices
	vector<MatrixXd> nabla_w(num_layers - 1);
	vector<VectorXd> nabla_b(num_layers - 1);
	// calculate delta for last layer
	VectorXd res = sigmoidPrime(z_vectors[num_layers - 1]);
	res.transposeInPlace();
	VectorXd delta = costDerivative(activations[activations.size() - 1], y) * res;
	// fill in the bias vector and weight matrix for last layer
	nabla_b[num_layers - 1] = delta;
	nabla_w[num_layers - 1] = activations[num_layers - 2] * delta;
	for (int i = num_layers - 2; i > 0; i--) {
		// compute the backpropagation equations and fill in the gradient matrices
		MatrixXd weight_matrix = weight_matrices[i + 1];
		weight_matrix.transposeInPlace();
		VectorXd z_vector = z_vectors[i];
		VectorXd temp = (weight_matrix * sigmoidPrime(z_vector)) * delta;
		delta = temp;

		nabla_b[i] = delta;
		nabla_w[i] = delta * activations[i - 1].transpose();
	}
	return make_tuple(nabla_w, nabla_b);
}