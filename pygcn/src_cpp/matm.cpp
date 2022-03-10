#include "../include/matm.h"
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

// TOP ORDERS ARE 
//  I K J
//  z x y
vector<vector<double>> matrix_multi(vector<vector<double>> input, vector<vector<double>> weight, vector<vector<double>> adj){
    vector<vector<double>> support_best;
    support_best.resize(input.size(), std::vector<double>(weight[0].size()));
    for (size_t i = 0; i < input.size(); i++){
        for (size_t k = 0; k < weight.size(); k++){
            for (size_t j = 0; j < weight[0].size(); j++){
                support_best[i][j] += input[i][k] * weight[k][j];
            }
        }
    }
    vector<vector<double>> output_best;
    output_best.resize(adj.size(), std::vector<double>(support_best[0].size()));
    for (size_t z = 0; z < support_best.size(); z++){
        for (size_t x = 0; x < adj.size(); x++){
            for (size_t y = 0; y < support_best[0].size(); y++){
                output_best[x][y] += adj[x][z] * support_best[z][y];
            }
        }
    }
    py::cast(output_best);
    return output_best;

    // ***************Below is used for finding best loop order*************************************************
    
//     using std::chrono::high_resolution_clock;
//     using std::chrono::duration_cast;
//     using std::chrono::duration;
//     using std::chrono::milliseconds;
//     string filename("looporder1.csv");
//     ofstream file_out;
//     file_out.open(filename, std::ios_base::app);
//     // I, j k
//     auto t1 = high_resolution_clock::now();
//     vector<vector<double>> support;
//     support.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t i = 0; i < input.size(); i++){
//         for (size_t j = 0; j < weight[0].size(); j++){
//             for (size_t k = 0; k < weight.size(); k++){
//                 support[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t2 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double = t2 - t1;
//     file_out << ms_double.count() << ", ";

//     // I, k, j
//     auto t3 = high_resolution_clock::now();
//     vector<vector<double>> support2;
//     support2.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t i = 0; i < input.size(); i++){
//         for (size_t k = 0; k < weight.size(); k++){
//             for (size_t j = 0; j < weight[0].size(); j++){
//                 support2[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t4 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double2 = t4 - t3;
//     file_out << ms_double2.count() << ", ";

//     // j i k
//     auto t5 = high_resolution_clock::now();
//     vector<vector<double>> support3;
//     support3.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t j = 0; j < weight[0].size(); j++){
//         for (size_t i = 0; i < input.size(); i++){
//             for (size_t k = 0; k < weight.size(); k++){
//                 support3[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t6 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double3 = t6 - t5;
//     file_out << ms_double3.count() << ", ";

//     // j k i
//     auto t7 = high_resolution_clock::now();
//     vector<vector<double>> support4;
//     support4.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t j = 0; j < weight[0].size(); j++){
//         for (size_t k = 0; k < weight.size(); k++){
//             for (size_t i = 0; i < input.size(); i++){
//                 support4[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t8 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double4 = t8 - t7;
//     file_out << ms_double4.count() << ", ";

//     // k i j
//     auto t9 = high_resolution_clock::now();
//     vector<vector<double>> support5;
//     support5.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t k = 0; k < weight.size(); k++){
//         for (size_t i = 0; i < input.size(); i++){
//             for (size_t j = 0; j < weight[0].size(); j++){
//                 support5[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t10 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double5 = t10 - t9;
//     file_out << ms_double5.count() << ", ";

//     // k j i
//     auto t11 = high_resolution_clock::now();
//     vector<vector<double>> support6;
//     support6.resize(input.size(), std::vector<double>(weight[0].size()));
//     for (size_t k = 0; k < weight.size(); k++){
//         for (size_t i = 0; i < input.size(); i++){
//             for (size_t j = 0; j < weight[0].size(); j++){
//                 support6[i][j] += input[i][k] * weight[k][j];
//             }
//         }
//     }
//     auto t12 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double6 = t12 - t11;
//     file_out << ms_double6.count() << ",\n";


//  /// **************************************** start of OUTPUT****************************************
//     string filename2("looporder2.csv");
//     ofstream file_out2;
//     file_out2.open(filename2, std::ios_base::app);

//     // x  y  z
//     auto t13 = high_resolution_clock::now();
//     vector<vector<double>> output;
//     output.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t x = 0; x < adj.size(); x++){
//         for (size_t y = 0; y < support[0].size(); y++){
//             for (size_t z = 0; z < support.size(); z++){
//                 output[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t14 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double7 = t14 - t13;
//     file_out2 << ms_double7.count() << ",";
    
//     // x z y
//     auto t15 = high_resolution_clock::now();
//     vector<vector<double>> output2;
//     output2.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t x = 0; x < adj.size(); x++){
//         for (size_t z = 0; z < support.size(); z++){
//             for (size_t y = 0; y < support[0].size(); y++){
//                 output2[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t16 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double8 = t16 - t15;
//     file_out2 << ms_double8.count() << ",";

//     // y x z
//     auto t17 = high_resolution_clock::now();
//     vector<vector<double>> output3;
//     output3.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t y = 0; y < support[0].size(); y++){
//         for (size_t x = 0; x < adj.size(); x++){
//             for (size_t z = 0; z < support.size(); z++){
//                 output3[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t18 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double9 = t18 - t17;
//     file_out2 << ms_double9.count() << ",";

//     // y z x
//     auto t19 = high_resolution_clock::now();
//     vector<vector<double>> output4;
//     output4.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t y = 0; y < support[0].size(); y++){
//         for (size_t z = 0; z < support.size(); z++){
//             for (size_t x = 0; x < adj.size(); x++){
//                 output4[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t20 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double10 = t20 - t19;
//     file_out2 << ms_double10.count() << ",";

//     // z x y
//     auto t21 = high_resolution_clock::now();
//     vector<vector<double>> output5;
//     output5.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t z = 0; z < support.size(); z++){
//         for (size_t x = 0; x < adj.size(); x++){
//             for (size_t y = 0; y < support[0].size(); y++){
//                 output5[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t22 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double11 = t22 - t21;
//     file_out2 << ms_double11.count() << ",";

//     // z y x
//     auto t23 = high_resolution_clock::now();
//     vector<vector<double>> output6;
//     output6.resize(adj.size(), std::vector<double>(support[0].size()));
//     for (size_t z = 0; z < support.size(); z++){
//         for (size_t y = 0; y < support[0].size(); y++){
//             for (size_t x = 0; x < adj.size(); x++){
//                 output6[x][y] += adj[x][z] * support[z][y];
//             }
//         }
//     }
//     auto t24 = high_resolution_clock::now();
//     duration<double, std::milli> ms_double12 = t24 - t23;
//     file_out2 << ms_double12.count() << ",\n";
    // cout << "\n\narray size:" << new_array.size() << ", " << new_array[0].size() << endl;

    //******************************************************************************* check for 0s

    // py::cast(output);
    // return output;
}