#ifndef FEATURE_MATCHING_HPP
#define FEATURE_MATCHING_HPP

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

// Helper function to extract keypoints and descriptors from an image using the provided SIFT detector
bool extractFeatures(const cv::Ptr<cv::SIFT> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

// Accumulates the distances for each keypoint in the reference frame for a single subsequent frame
void accumulateDistancesForFrame(const cv::Mat &referenceDescriptors, const std::vector<cv::KeyPoint> &referenceKeypoints, const cv::Mat &frame, const cv::Ptr<cv::SIFT> &detector, cv::BFMatcher &matcher, std::vector<double> &distances);

// Computes the accumulated distances for each keypoint in the reference frame across all subsequent frames
// I want to know how much a keypoint move
std::vector<double> computeReferenceKeypointDistances(const std::vector<cv::Mat> &images, const cv::Ptr<cv::SIFT> &detector, const std::vector<cv::KeyPoint> &referenceKeypoints, const cv::Mat &referenceDescriptors);

#endif // FEATURE_MATCHING_HPP