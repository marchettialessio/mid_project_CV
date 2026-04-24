#ifndef HYBRID_APPROACH_HPP
#define HYBRID_APPROACH_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include "optical_flow.hpp"


template<typename T>
void updateTrackedKeypoints(const cv::Mat &firstFrameDescriptors, const std::vector<cv::KeyPoint> &firstFrameKeypoints, const cv::Mat &currentFrameGray, const cv::Ptr<T> &detector, std::vector<cv::Point2f> &verifiedFirstFrameKP, OpticalFlow &opticalFlow);

#endif // HYBRID_APPROACH_HPP

