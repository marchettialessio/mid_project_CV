#include <opencv2/features2d.hpp>
#include "feature_matching.hpp"
#include "utils.hpp"
#include "visualization.hpp"
#include <iostream>
#include <vector>

int main()
{
    const std::string path = "../../resources/data/car";
    std::vector<cv::Mat> images;

    loadImages(path, images);

    if (images.empty())
    {
        std::cerr << "Error: no valid images found." << std::endl;
        return -1;
    }

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypointsFirstFrame;
    cv::Mat descriptorsFirstFrame;

    if (!extractFeatures(detector, images[0], keypointsFirstFrame, descriptorsFirstFrame))
    {
        std::cerr << "Error: no keypoints detected in the first frame." << std::endl;
        return -1;
    }

    std::vector<double> distances = computeReferenceKeypointDistances(images, detector, keypointsFirstFrame, descriptorsFirstFrame);

    runThresholdViewer(images[0], keypointsFirstFrame, distances);

    return 0;
}