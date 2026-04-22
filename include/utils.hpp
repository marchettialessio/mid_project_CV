#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

void loadImages(const std::string& directory, std::vector<cv::Mat>& images);

#endif // UTILS_HPP