#pragma once
#include <string>
#include <sstream>

using namespace std;

double ToDouble(string str) {
	stringstream ss(str);
	int val;
	ss >> val;
	return (double)val;
}

// This function reads the csv file and returns a tuple consisting of the predictors X and the output y.
vector<tuple<vector<double>, vector<double>>>& readCsv(string filename) {
	ifstream file(filename);
	string line;
	static vector<tuple<vector<double>, vector<double>>> dataset;
	// get next line and split into tokens
	int line_count = 0;
	cout << "Reading file" << endl;
	while (getline(file, line)) {
		if (line_count == 0) {
			line_count = 1;
			continue;
		}
		string token;
		stringstream line_stream(line);
		vector<double> X, y(10);
		int cell_count = 0;
		while (getline(line_stream, token, ',')) {
			if (cell_count == 0) {
				for (int i = 0; i < 10; i++) {
					if ((double)i + 1.0 == ToDouble(token))
						y[i] = 1.0;
					else
						y[i] = 0.0;
				}
				cell_count = 1;
				continue;
			}
			X.push_back(ToDouble(token));
		}
		dataset.push_back(make_tuple(X, y));
	}
	cout << "Reading complete" << endl;
	return dataset;
}

