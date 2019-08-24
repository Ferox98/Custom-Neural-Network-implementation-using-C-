# Custom-Neural-Network-implementation-using-C-
This C++ project uses Eigen (a lightweight Linear Algebra library) to implement a Neural Network from scratch. The latest version of Eigen can be found and downloaded at http://eigen.tuxfamily.org/index.php?title=Main_Page.

Example usage:
```c++
// Define the network architecture. In this case, a network with 748 input layers, 2 16-neuron hidden layers and an output layer with 10 // neurons.
vector<int> layers = {748, 16, 16, 10};
// Initialize the network with the architecture as a parameter
Network net(layers);
// include the dataset to train the network on. For this example, the MNIST dataset will be used.
Data d = readCsv("mnist_train.csv");
// Train the network using Stochastic Gradient Descent. The SGD function takes the number of epochs, the mini-batch size, the training
// data and the learning rate as a parameter
net.SGD(1, 16, d, 3.0);
// Now that the training has finished, test the prediction accuracy with a test dataset
d = readCsv("mnist_test.csv");
vector<double> X = get<0>(d[0]);
int output = net.predict(X);
cout << output << endl;
cin.ignore();
```
