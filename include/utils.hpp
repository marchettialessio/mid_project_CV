#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>


// Function to load all images from a specified directory
void loadImages(const std::string& directory, std::vector<cv::Mat>& images);

// Function to filter keypoints based on a distance threshold
std::vector<cv::KeyPoint> filterKeypointsAboveThreshold(const std::vector<cv::KeyPoint> &keypoints, const std::vector<double> &distances, int threshold);


#endif // UTILS_HPP