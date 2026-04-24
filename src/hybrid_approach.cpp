#include "hybrid_approach.hpp"
#include "process_images.hpp"
#include <opencv2/features2d.hpp>

template <typename T>
void updateTrackedKeypoints(cv::Mat firstFrameDescriptors, std::vector<cv::KeyPoint> firstFrameKeypoints, cv::Mat currentFrameGray, const cv::Ptr<T> &detector, std::vector<cv::Point2f> &verifiedFirstFrameKP, OpticalFlow &opticalFlow)
{

    // i want to keep all the keypoints that survived so far, also if they are no longer tracked
    if (verifiedFirstFrameKP.empty())
    {
        // in first iteration, we just take the surviving keypoints from the optical flow
        verifiedFirstFrameKP = opticalFlow.getSurvivingKP();
    }
    else
    {
        // in the next iterations, i want to make to add new points that were matched and survived the optical flow (I do a union)
        std::vector<cv::Point2f> newKP = opticalFlow.getSurvivingKP();
        for (const auto &pt : newKP)
        {
            bool found = false;
            for (const auto &v_pt : verifiedFirstFrameKP)
            {
                // Se la distanza tra i punti è minima, li consideriamo identici
                if (cv::norm(v_pt - pt) < 1.0)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                verifiedFirstFrameKP.push_back(pt);
            }
        }
    }
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> currentFrameKeypoints;
    cv::Mat currentFrameDescriptors;

    if (!extractFeatures(detector, currentFrameGray, keypoints, currentFrameDescriptors))
    {
        return;
    }

    cv::KeyPoint::convert(keypoints, currentFrameKeypoints); // Convert to point in 2D

    cv::BFMatcher matcher(cv::NORM_L2, true);
    std::vector<cv::DMatch> matches;
    matcher.match(firstFrameDescriptors, currentFrameDescriptors, matches);

    // i've done matches, now i want to update the nuew keypoints
    std::vector<cv::Point2f> currKP;
    std::vector<cv::Point2f> survivedKP;
    for (auto match : matches)
    {
        // Accedi ai keypoint tramite gli indici del match
        const auto &kpCurrent = currentFrameKeypoints[match.trainIdx];
        const auto &kpFirst = firstFrameKeypoints[match.queryIdx];

        // Inserisci i punti creando oggetti Point2f al volo dalle coordinate x e y
        currKP.push_back(cv::Point2f(kpCurrent.x, kpCurrent.y));
        survivedKP.push_back(cv::Point2f(kpFirst.pt.x, kpFirst.pt.y));
    }
    opticalFlow.setCurrKP(currKP);
    opticalFlow.setSurvivingKP(survivedKP);
}
template void updateTrackedKeypoints<cv::SIFT>(cv::Mat firstFrameDescriptors, std::vector<cv::KeyPoint> firstFrameKeypoints, cv::Mat currentFrameGray, const cv::Ptr<cv::SIFT> &detector, std::vector<cv::Point2f> &verifiedFirstFrameKP, OpticalFlow &opticalFlow);
