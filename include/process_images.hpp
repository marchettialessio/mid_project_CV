#ifndef PROCESS_IMAGES_HPP
#define PROCESS_IMAGES_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>


template<typename T>
void processImageSequence(std::vector<cv::Mat> images, cv::Ptr<T> detector, float minMotion);

template <typename T>
bool extractFeatures(const cv::Ptr<T> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
#endif // PROCESS_IMAGES_HPP
