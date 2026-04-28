/*
@author: Nicolò Spuri
*/
#include "optical_flow.hpp"
#include <opencv2/video.hpp>

void OpticalFlow::calcOpticalFlowIteration(cv::Mat grayFrame, cv::Mat oldGray, cv::Mat &frame, cv::Mat &mask, std::vector<cv::Scalar> &colors)
{
    std::vector<cv::Point2f> newKP;
    std::vector<uchar> status;
    std::vector<float> err;
    // TODO: understand criteria
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(oldGray, grayFrame, this->currKP, newKP, status, err, cv::Size(15, 15), 2, criteria);
    std::vector<cv::Point2f> moving_pts, good_new;

    // Scan the vectors of keypoints
    // scan from the end, i want to erase keypoints that don't survive
    for (uint j = 0; j < currKP.size(); j++)
    {
        // Select good points, status == 1 if flow has been found
        if (status[j] == 1)
        {
            float d = cv::norm(newKP[j] - this->currKP[j]); // Euclidean distance
            // TODO: impostare che se una feature si è mossa un tot di volte è definitiva, perché ad esempio nella frog alcune feature finiscono fuori dall'immagine e vengono perse
            if (d >= this->minMotion)
            { // if the point moved more or equal than minMotion -> track the flow
                moving_pts.push_back(this->survivingKP[j]);
                good_new.push_back(newKP[j]);

                // Draw the tracks
                cv::line(mask, newKP[j], this->currKP[j], colors[j % colors.size()], 2);
                circle(frame, newKP[j], 5, colors[j % colors.size()], -1);
            }
        }
    }

    // update survivingKP and newKP for the next iteration
    this->survivingKP = moving_pts;
    this->currKP = good_new;
}