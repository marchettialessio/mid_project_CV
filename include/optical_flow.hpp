#ifndef OPTICAL_FLOW_HPP
#define OPTICAL_FLOW_HPP

#include <opencv2/features2d.hpp>
#include <vector>

class OpticalFlow
{
public:
    OpticalFlow(std::vector<cv::Point2f> currKP, std::vector<cv::Point2f> survivingKP, float minMotion) : currKP(currKP), survivingKP(survivingKP), minMotion(minMotion)
    {}

    void calcOpticalFlowIteration(cv::Mat grayFrame, cv::Mat oldGray, cv::Mat& frame, cv::Mat& mask, std::vector<cv::Scalar>& colors); 

    std::vector<cv::Point2f> getSurvivingKP() const {
        return survivingKP;
    }
    void setSurvivingKP(const std::vector<cv::Point2f>& newSurvivingKP) {
        survivingKP = newSurvivingKP;
    }

    void setCurrKP(const std::vector<cv::Point2f>& newCurrKP) {
        currKP = newCurrKP;
    }

    // Function to calculate optical flow and filter keypoints based on motion
private:
    // Optical flow parameters
    float minMotion;                      // Minimum motion threshold for filtering keypoints
    std::vector<cv::Point2f> currKP;      // Current keypoints
    std::vector<cv::Point2f> survivingKP; // Keypoints that survive the motion filter
};

#endif // OPTICAL_FLOW_HPP