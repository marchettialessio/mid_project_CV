#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

void runThresholdViewer(const cv::Mat &referenceImage, const std::vector<cv::KeyPoint> &referenceKeypoints, const std::vector<double> &distances);

#endif // VISUALIZATION_HPP