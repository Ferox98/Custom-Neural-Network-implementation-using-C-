#pragma once
#include <string>
#include <sstream>

using namespace std;

// This function reads the csv file and returns a tuple consisting of the predictors X and the output y.
static vector<tuple<vector<double>, vector<double>>>& readCsv(string filename) {
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
					if ((double)i + 1.0 == atof(token.c_str()))
						y[i] = 1.0;
					else
						y[i] = 0.0;
				}
				cell_count = 1;
				continue;
			}
			// push back normalized value
			X.push_back(atof(token.c_str()) / 255.0);
		}
		dataset.push_back(make_tuple(X, y));
	}
	cout << "Reading complete" << endl;
	return dataset;
}

