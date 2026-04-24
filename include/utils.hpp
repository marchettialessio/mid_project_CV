#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>


// Function to load all images from a specified directory
void loadImages(const std::string& directory, std::vector<cv::Mat>& images);

std::vector<cv::Scalar> generateRandomColors(int num_colors);




#endif // UTILS_HPP