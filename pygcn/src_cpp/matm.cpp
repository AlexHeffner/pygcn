#include "../include/matm.h"
#include <vector>
using namespace std;

vector<vector<double>> matrix_multi(vector<vector<double>> input, vector<vector<double>> weight, vector<vector<double>> adj){
    vector<vector<double>> support;
    support.resize(input.size(), std::vector<double>(weight[0].size()));
    for (size_t i = 0; i < input.size(); i++){
        for (size_t j = 0; j < weight[0].size(); j++){
            for (size_t k = 0; k < weight.size(); k++){
                support[i][j] += input[i][k] * weight[k][j];
            }
        }
    }
    vector<vector<double>> output;
    output.resize(adj.size(), std::vector<double>(support[0].size()));
    for (size_t x = 0; x < adj.size(); x++){
        for (size_t y = 0; y < support[0].size(); y++){
            for (size_t z = 0; z < support.size(); z++){
                output[x][y] += adj[x][z] * support[z][y];
            }
        }
    }
    // cout << "\n\narray size:" << new_array.size() << ", " << new_array[0].size() << endl;

    py::cast(output);
    return output;
}