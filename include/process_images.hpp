#ifndef PROCESS_IMAGES_HPP
#define PROCESS_IMAGES_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>


// Process an image sequence. If outBox is provided, the final bounding box
// computed for the tracked object will be written into it. If `show` is false
// no GUI windows will be displayed (useful for batch evaluation).
void processImageSequence(std::vector<cv::Mat> images, cv::Ptr<cv::Feature2D> detector, float minMotion, cv::Rect* outBox = nullptr, bool show = true);
// Extract features from an image using the provided feature detector
bool extractFeatures(const cv::Ptr<cv::Feature2D> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

#endif // PROCESS_IMAGES_HPP
