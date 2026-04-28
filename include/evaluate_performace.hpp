/*
@author: Sirio Trentin
*/
#ifndef EVALUATE_PERFORMANCE_HPP
#define EVALUATE_PERFORMANCE_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include "evaluation_results.hpp"

EvaluationResults evaluatePerformance(const std::string& directory_progetto);

#endif