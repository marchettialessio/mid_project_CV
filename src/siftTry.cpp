#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

int main() {
    cv::Mat img = cv::imread("../resources/data/frog/0000.png");
    cv::imshow("Source image", img);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output);
    cv::imshow("Features", output);
    cv::waitKey();

    return 0;
}