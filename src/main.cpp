#include <opencv2/features2d.hpp>
#include "feature_matching.hpp"
#include "utils.hpp"
#include "visualization.hpp"
#include <iostream>
#include <vector>
#include <filesystem>

#include "tracking.hpp"
#include <opencv2/highgui.hpp>

int main()
{
    const std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) / "resources" / "data" / "frog";
    std::vector<cv::Mat> images;

    loadImages(path.string(), images);

    if (images.empty())
    {
        std::cerr << "Error: no valid images found." << std::endl;
        return -1;
    }

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create(0, 6);
    std::vector<cv::KeyPoint> keypointsFirstFrame;
    cv::Mat descriptorsFirstFrame;

    if (!extractFeatures(detector, images[0], keypointsFirstFrame, descriptorsFirstFrame))
    {
        std::cerr << "Error: no keypoints detected in the first frame." << std::endl;
        return -1;
    }

    // std::vector<double> distances = computeReferenceKeypointDistances(images, detector, keypointsFirstFrame, descriptorsFirstFrame);

    cv::BFMatcher matcher(cv::NORM_L2, false);
    Tracking tracking(keypointsFirstFrame, descriptorsFirstFrame, 70);
    for (size_t i = 1; i < images.size(); i++)
    {

        tracking.updateTracking(images[i], detector, matcher);
    }
     //runThresholdViewer(images[0], keypointsFirstFrame, distances);

    cv::Mat output;
    drawFilteredKeypoints(images[0], tracking.returnFeaturesCurrent(keypointsFirstFrame), output);
    cv::imshow("prova", output);
    int key = cv::waitKey();
    return 0;
}