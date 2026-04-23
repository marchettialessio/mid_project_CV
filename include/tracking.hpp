#include <vector>
#include <opencv2/features2d.hpp>


class Tracking
{
public:
    Tracking(std::vector<cv::KeyPoint> keypoints_reference_frame, cv::Mat descriptors_reference_frame, int total_distance_threshold);

    // Update tracking status based on the new frame and refresh tracking status every few frames
    void updateTracking(const cv::Mat &frame, const cv::Ptr<cv::SIFT> &detector, cv::BFMatcher &matcher);

    std::vector<cv::KeyPoint> returnFeaturesCurrent(const std::vector<cv::KeyPoint> &keypoints);

private:
    std::vector<cv::KeyPoint> keypoints_reference_frame;
    cv::Mat descriptors_reference_frame;
    std::vector<cv::KeyPoint> keypoints_frame;
    cv::Mat descriptors_frame;
    std::vector<bool> active_tracking;
    std::vector<double> total_displacements;
    std::vector<int> tracked_features_indices; // Stores the indices of the currently tracked features in the current frame
    int tracking_status_refresh_rate = 5; // Refresh tracking status every 5 frames
    int current_tracking_frame_count = 0; // Counter to keep track of frames for refreshing tracking status
    bool is_first_frame = true; // Flag to indicate if it's the first frame being processed
    int total_distance_threshold; // How much a keypoint has to moove to keep tracking
};