/*
@author: Alessio Marchetti
*/
#ifndef EVALUATION_RESULTS_HPP
#define EVALUATION_RESULTS_HPP
#include <vector>

struct EvaluationResults {
    std::vector<std::string> categories;
    std::vector<double> ious;
    std::vector<std::string> imagePaths;
    int correct_folders;
    int total_folders;
    double meanIoU;
    double accuracy;
};
#endif