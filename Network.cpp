#include "Network.h"

Network::Network(vector<int>& layers) {
	m_layers = layers;
	// initialize the weight, and bias matrices
	cout << "Initializing weight and bias matrices" << endl;
	weight_matrices = vector<MatrixXd>(layers.size());
	bias_vectors = vector<VectorXd>(layers.size());

	for (int i = 0; i < layers.size(); i++) {
		if (i != layers.size() - 1)
			weight_matrices[i] = MatrixXd::Random(layers[i+1], layers[i]);
		if (i != 0) {
			// input layer has no bias vector
			bias_vectors[i] = VectorXd::Random(layers[i]); // a column vector with specified dimensions and -1 to 1 randomly initialized values
		}
	}
	std::cout << "All weight matrices and bias vectors have been successfully created" << std::endl;
}

VectorXd Network::sigmoidPrime(VectorXd& vec) {
	VectorXd activation(vec.rows());
	for (int i = 0; i < vec.rows(); i++)
		activation(i) = sigmoid(vec(i)) * (1 - sigmoid(vec(i)));
	return activation;
}

VectorXd Network::costDerivative(VectorXd& output, vector<double>& y) {
	// return a vector of partial derivatives partial Cx / partial a(k) 
	VectorXd vec(output.rows());

	for (int i = 0; i < output.rows(); i++) {
		vec(i) = (output(i) - y[i]);
	}
	return vec;
}

void Network::SGD(int epochs, int mini_batch_size, Data& training_data, double eta) {
	// This function performs stochastic gradient descent on the network using specified mini-batch size
	cout << "Starting training" << endl;
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
	cout << "Training complete" << endl;
}

void Network::updateNetwork(Data& mini_batch) {
	// retrieves gradients for the weights and biases and updates the network with them
	int len = mini_batch.size();
	for (auto& _tuple : mini_batch) {
		vector<double> X = get<0>(_tuple);
		vector<double> y = get<1>(_tuple);
		tuple<vector<MatrixXd>, vector<VectorXd>> matrices = backpropagate(X, y);
		vector<MatrixXd> weight_gradients = get<0>(matrices);
		vector<VectorXd> bias_gradients = get<1>(matrices);

		for (int i = 0; i < weight_gradients.size() - 1; i++) {
			weight_matrices[i] -= (eta / (double)len) * weight_gradients[i];
		}

		for (int i = 1; i < bias_gradients.size(); i++) {
			bias_vectors[i] -= (eta / (double)len) * bias_gradients[i];
		}
	}
}

tuple<vector<MatrixXd>, vector<VectorXd>> Network::backpropagate(vector<double>& x, vector<double>& y) {
	// feed-forward
	VectorXd activation(x.size());
	vector<VectorXd> activations;
	for (int i = 0; i < x.size(); i++)
		activation(i) = x[i];
	activations.push_back(activation);
	vector<VectorXd> z_vectors(m_layers.size());
	
	for (int i = 1; i < m_layers.size(); i++) {
		VectorXd z = weight_matrices[i - 1] * activation + bias_vectors[i];
		activation = VectorXd(m_layers[i]);
		for (int j = 0; j < z.rows(); j++) {
			double sigmoid_x = sigmoid(z[j]);
			activation[j] = sigmoid_x;
		}
		z_vectors[i] = z;
		activations.push_back(activation);
	}
	// perform backward pass
	// initialize gradient matrices
	vector<MatrixXd> nabla_w = vector<MatrixXd>(m_layers.size());
	vector<VectorXd> nabla_b = vector<VectorXd>(m_layers.size());
	// calculate delta for last layer. Note that delta is the Hadamard product of the cost derivative partial c / partial a and the sigmoid prime of the z vector
	ArrayXXd res = (ArrayXXd) sigmoidPrime(z_vectors[m_layers.size() - 1]);
	ArrayXXd cost_der = (ArrayXXd)costDerivative(activations[activations.size() - 1], y);
	ArrayXXd delta = cost_der * res;
	VectorXd vec_delta = (VectorXd)delta;
	// fill in the bias vector and weight matrix for last layer
	nabla_b[nabla_b.size() - 1] = delta;
	RowVectorXd activation_transpose = (RowVectorXd)activations[activations.size() - 2];
	nabla_w[nabla_w.size() - 2] = vec_delta * activation_transpose;

	for (int i = m_layers.size() - 2; i > 0; i--) {
		// compute the backpropagation equations and fill in the gradient matrices
		MatrixXd weight_matrix = weight_matrices[i];
		weight_matrix.transposeInPlace();
		VectorXd z_vector = z_vectors[i];
		VectorXd sigmoid_z = sigmoidPrime(z_vector);
		VectorXd temp = (weight_matrix * vec_delta);
		ArrayXXd z_array = (ArrayXXd)sigmoid_z;
		delta = (ArrayXXd)temp * z_array;
		vec_delta = (VectorXd)delta;
		nabla_b[i] = vec_delta;
		activation_transpose = activations[i-1].transpose();
		nabla_w[i - 1] = vec_delta * activation_transpose;
	}
	return make_tuple(nabla_w, nabla_b);
}

void Network::saveNetwork(string filename) {
	cout << "Saving Network" << endl;
	ofstream file;
	file.open(filename);
	for (auto& i : m_layers)
		file << i << ",";
	file << "\n";
	
	for (auto& matrix : weight_matrices) {
		for (int row_count = 0; row_count < matrix.rows(); row_count++) {
			for (int col_count = 0; col_count < matrix.cols(); col_count++) {
				file << matrix(row_count, col_count);
				file << " \t";
			}
			file << "\n";
		}
	}

	for (auto& vector : bias_vectors) {
		for (int row_count = 0; row_count < vector.rows(); row_count++) {
			file << vector(row_count) << "\n";
		}
	}
	file.close();
	cout << "Network successfully saved in file" << endl;
}

void Network::loadNetwork(string filename) {
	cout << "Loading network from file" << endl;
	ifstream file;
	file.open(filename);
	// read network architecture
	string line;
	getline(file, line, '\n');
	string input;
	stringstream linestream(line);
	vector<int> layers;
	while (getline(linestream, input, ',')) {
		layers.push_back(atoi(input.c_str()));
	}
	// Now get the weight matrices and bias vectors
	int layer = 0;
	vector<MatrixXd> weight_matrices;
	vector<VectorXd> bias_vectors;

	while (layer < layers.size() - 1) {
		// get the dimensions of the matrix
		MatrixXd weight_matrix = MatrixXd(layers[layer + 1], layers[layer]);
		int col_count = 0, row_count = 0;
		while (row_count < weight_matrix.rows()) {
			getline(file, line, '\n');
			linestream = stringstream(line);
			while (getline(linestream, input, '\t')) {
				weight_matrix(row_count, col_count) = atof(input.c_str());
				col_count++;
				if (col_count == weight_matrix.cols())
					break;
			}
			row_count++;
			col_count = 0;
		}
		weight_matrices.push_back(weight_matrix);
		layer++;
	}

	layer = 0;
	while (layer < layers.size() - 1) {
		VectorXd bias_vector(layers[layer + 1]);
		int row_count = 0;
		while (row_count < bias_vector.rows()) {
			getline(file, line, '\n');
			bias_vector(row_count) = atof(line.c_str());
			row_count++;
		}
		bias_vectors.push_back(bias_vector);
		layer++;
	}
	// initialize network with obtained parameters
	this->weight_matrices = weight_matrices;
	this->bias_vectors = bias_vectors;
	this->m_layers = layers;

	file.close();
}

int Network::predict(vector<double>& x) {
	// feed-forward
	vector<VectorXd> activations = vector<VectorXd>(m_layers.size());
	VectorXd activation(x.size());
	for (int i = 0; i < x.size(); i++)
		activation(i) = x[i];
	activations[0] = activation;
	for (int i = 1; i < m_layers.size(); i++) {
		VectorXd z = weight_matrices[i - 1] * activations[i - 1] + bias_vectors[i];
		activation = VectorXd(m_layers[i]);
		for(int j = 0; j < z.rows(); j++) {
			activation(j) = sigmoid(z[j]);
		}
		activations[i] = activation;
	}
	// get the max value from the last layer of the activation matrix
	VectorXd output = activations[activations.size() - 1];
	double max_val = output(0);
	int max_idx = 0;
	for (int i = 0; i < output.rows(); i++) {
		if (output(i) > max_val) {
			max_val = output(i);
			max_idx = i;
		}
	}
	return max_idx;
}