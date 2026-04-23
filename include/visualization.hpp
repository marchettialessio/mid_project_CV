#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include <vector>
#include <opencv2/features2d.hpp>

void runThresholdViewer(const cv::Mat &referenceImage, const std::vector<cv::KeyPoint> &referenceKeypoints, const std::vector<double> &distances);

void drawFilteredKeypoints(const cv::Mat &image, const std::vector<cv::KeyPoint> &filteredKeypoints, cv::Mat &output);

#endif // VISUALIZATION_HPP