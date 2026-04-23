#include "tracking.hpp"
#include "feature_matching.hpp"
#include <algorithm>

Tracking::Tracking(std::vector<cv::KeyPoint> keypoints_reference_frame, cv::Mat descriptors_reference_frame, int total_distance_threshold)
{
    this->keypoints_reference_frame = keypoints_reference_frame;
    this->descriptors_reference_frame = descriptors_reference_frame;
    this->active_tracking.resize(keypoints_reference_frame.size(), false);
    this->total_displacements.resize(keypoints_reference_frame.size(), 0.0);
    this->tracked_features_indices.resize(keypoints_reference_frame.size(), -1);
    this->total_distance_threshold = total_distance_threshold;
}

void Tracking::updateTracking(const cv::Mat &frame, const cv::Ptr<cv::SIFT> &detector, cv::BFMatcher &matcher)
{
    // i extract the features and desctiptor from current frame
    if (!extractFeatures(detector, frame, this->keypoints_frame, this->descriptors_frame))
    {
        return;
    }

    std::vector<cv::DMatch> matches;
    matcher.match(this->descriptors_reference_frame, this->descriptors_frame, matches);

    const bool refresh_tracking_status = (this->current_tracking_frame_count + 1 == this->tracking_status_refresh_rate);

    for (const auto match : matches)
    {
        // 1: i calculate the index i want to update
        // queryIdx => reference frame, trainIdx => current frame
        // the index that i want to update
        int updateIndex = -1;
        if (this->is_first_frame)
        {
            // I can do this directly only on first iteration
            updateIndex = match.queryIdx;
            // i activate the right features
            this->active_tracking[updateIndex] = true;
        }
        else
        {
            // Find the tracked slot among active features only, to avoid stale indices from inactive ones.
            for (size_t i = 0; i < this->tracked_features_indices.size(); ++i)
            {
                if (this->active_tracking[i] && this->tracked_features_indices[i] == match.queryIdx)
                {
                    updateIndex = static_cast<int>(i);
                    break;
                }
            }

            if (updateIndex == -1)
            {
                // i'm not tracking this feature
                continue;
            }
        }

        // 2: i want to update only if i'm tracking
        if (!this->active_tracking[updateIndex])
        {
            continue;
        }

        // 3: I'm tracking this feature!! => update indices and distance
        // now I'm keeping track in the correct index position
        this->tracked_features_indices[updateIndex] = match.trainIdx;
        // distance between reference and current frame
        double distance = std::hypot(this->keypoints_reference_frame[match.queryIdx].pt.x - this->keypoints_frame[match.trainIdx].pt.x,
                                     this->keypoints_reference_frame[match.queryIdx].pt.y - this->keypoints_frame[match.trainIdx].pt.y);
        // total distance of the feature so far

        this->total_displacements[updateIndex] += distance;
    }

    if (refresh_tracking_status)
    {
        for (size_t i = 0; i < this->active_tracking.size(); ++i)
        {
            if (!this->active_tracking[i])
            {
                continue;
            }

            // Deactivate features that were not matched or moved too little in the refresh window.
            if (this->total_displacements[i] < this->total_distance_threshold)
            {
                this->active_tracking[i] = false;
            }
        }
    }

    if (this->is_first_frame)
    {
        this->is_first_frame = false;
    }

    // new reference frame
    this->keypoints_reference_frame = this->keypoints_frame;
    this->descriptors_reference_frame = this->descriptors_frame;

    if (refresh_tracking_status)
    {
        this->current_tracking_frame_count = 0;
    }
    else
    {
        this->current_tracking_frame_count++;
    }
}

std::vector<cv::KeyPoint> Tracking::returnFeaturesCurrent(const std::vector<cv::KeyPoint> &keypoints)
{
    std::vector<cv::KeyPoint> filteredKeypoints;
    filteredKeypoints.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        if (this->active_tracking[i])
        {
            filteredKeypoints.push_back(keypoints[i]);
        }
    }

    return filteredKeypoints;
}
