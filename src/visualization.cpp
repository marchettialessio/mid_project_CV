#include "visualization.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace
{
std::vector<cv::KeyPoint> filterKeypointsAboveThreshold(const std::vector<cv::KeyPoint> &keypoints, const std::vector<double> &distances, int threshold)
{
    std::vector<cv::KeyPoint> filteredKeypoints;
    filteredKeypoints.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        if (distances[i] > static_cast<double>(threshold))
        {
            filteredKeypoints.push_back(keypoints[i]);
        }
    }

    return filteredKeypoints;
}

void drawFilteredKeypoints(const cv::Mat &image, const std::vector<cv::KeyPoint> &filteredKeypoints, cv::Mat &output)
{
    cv::drawKeypoints(image, filteredKeypoints, output, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    if (filteredKeypoints.empty())
    {
        return;
    }

    std::vector<cv::Point2f> points;
    points.reserve(filteredKeypoints.size());

    for (const auto &kp : filteredKeypoints)
    {
        points.push_back(kp.pt);
    }

    cv::Rect bbox = cv::boundingRect(points);
    cv::rectangle(output, bbox, cv::Scalar(0, 0, 255), 2);
}
} // namespace

void runThresholdViewer(const cv::Mat &referenceImage, const std::vector<cv::KeyPoint> &referenceKeypoints, const std::vector<double> &distances)
{
    double maxDistance = *std::max_element(distances.begin(), distances.end());
    int maxThreshold = std::max(1, static_cast<int>(std::ceil(maxDistance)));

    const std::string windowName = "Keypoints above threshold";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::createTrackbar("Threshold", windowName, nullptr, maxThreshold);

    while (true)
    {
        int thresholdValue = cv::getTrackbarPos("Threshold", windowName);
        std::vector<cv::KeyPoint> filteredKeypoints = filterKeypointsAboveThreshold(referenceKeypoints, distances, thresholdValue);

        cv::Mat output;
        drawFilteredKeypoints(referenceImage, filteredKeypoints, output);

        cv::imshow(windowName, output);
        int key = cv::waitKey(30);

        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }
}