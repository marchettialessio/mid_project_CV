#include "feature_matching.hpp"
#include <cmath>


bool extractFeatures(const cv::Ptr<cv::SIFT> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    return !descriptors.empty() && !keypoints.empty();
}


void accumulateDistancesForFrame(const cv::Mat &referenceDescriptors, const std::vector<cv::KeyPoint> &referenceKeypoints, const cv::Mat &frame, const cv::Ptr<cv::SIFT> &detector, cv::BFMatcher &matcher, std::vector<double> &distances)
{
    std::vector<cv::KeyPoint> frameKeypoints;
    cv::Mat frameDescriptors;

    if (!extractFeatures(detector, frame, frameKeypoints, frameDescriptors))
    {
        return;
    }

    std::vector<cv::DMatch> matches;
    matcher.match(referenceDescriptors, frameDescriptors, matches);

    for (const auto &match : matches)
    {
        const cv::KeyPoint &keypoint1 = referenceKeypoints[match.queryIdx];
        const cv::KeyPoint &keypoint2 = frameKeypoints[match.trainIdx];

        // Compute the Euclidean distance between the matched keypoints and accumulate it
        double distance = std::hypot(keypoint1.pt.x - keypoint2.pt.x, keypoint1.pt.y - keypoint2.pt.y);
        distances[match.queryIdx] += distance;
    }
}

std::vector<double> computeReferenceKeypointDistances(const std::vector<cv::Mat> &images, const cv::Ptr<cv::SIFT> &detector, const std::vector<cv::KeyPoint> &referenceKeypoints, const cv::Mat &referenceDescriptors)
{
    std::vector<double> distances(referenceKeypoints.size(), 0.0);
    cv::BFMatcher matcher(cv::NORM_L2, true);

    for (size_t i = 1; i < images.size(); ++i)
    {
        accumulateDistancesForFrame(referenceDescriptors, referenceKeypoints, images[i], detector, matcher, distances);
    }

    return distances;
}