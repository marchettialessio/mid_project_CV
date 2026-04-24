#include "process_images.hpp"
#include "utils.hpp"
#include <opencv2/features2d.hpp>
#include "optical_flow.hpp"
#include "hybrid_approach.hpp"

template <typename T>
bool extractFeatures(const cv::Ptr<T> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    return !descriptors.empty() && !keypoints.empty();
}

template <typename T>
void processImageSequence(std::vector<cv::Mat> images, cv::Ptr<T> detector, float minMotion)
{

    std::vector<cv::Scalar> colors = generateRandomColors(100); // Generate random colors for drawing

    // Read initial frame and get grayscale
    cv::Mat firstFrame = images[0];
    cv::Mat firstFrameGray;
        cv::cvtColor(firstFrame, firstFrameGray, cv::COLOR_BGR2GRAY);

    // Detect keypoints using parameter detector
    std::vector<cv::Point2f> firstFrameKP;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat firstFrameDescriptors;
    if (!extractFeatures<T>(detector, firstFrameGray, keypoints, firstFrameDescriptors))
        return;                                     // Extract descriptors (not used in this code but can be useful for future extensions)
    cv::KeyPoint::convert(keypoints, firstFrameKP); // Convert to point in 2D

    std::cout << "Keypoints size: " << firstFrameKP.size() << std::endl;

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(firstFrame.size(), firstFrame.type());
    // Set frame and keypoints for the next iteration
    cv::Mat oldGray = firstFrameGray;
    std::vector<cv::Point2f> currKP = firstFrameKP;
    // this is the outut of optical flow => i want to know keypoints of first frame that survive during iterations
    //  To keep track of the initial position of the keypoints for drawing the rectangle
    std::vector<cv::Point2f> survivingKP = firstFrameKP;

    // TODO: se KP.size va a 0 break
    OpticalFlow opticalFlow(currKP, survivingKP, minMotion);
    std::vector<cv::Point2f> verifiedFirstFrameKP; // Here i store all the keypoints of the first frame that survive during an optimal flow pass
    int hybridApproachRate = 10;                   // Every N frames, we will redetect keypoints to handle occlusions and new features
    int currentFrameCounter = 0;                   // To keep track of the current frame number for the hybrid approach

    // Read and process images
    for (size_t i = 1; i < images.size(); i++)
    {
        // Read frame 'i' and get grayscale
        cv::Mat frame = images[i];
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // hybrid approach: every N frames, we redetect keypoints to handle occlusions and new features
        if (hybridApproachRate == currentFrameCounter + 1)
        {
            currentFrameCounter = 0; // reset counter
            mask = cv::Mat::zeros(firstFrame.size(), firstFrame.type());
            updateTrackedKeypoints<T>(firstFrameDescriptors, keypoints, oldGray, detector, verifiedFirstFrameKP, opticalFlow);
        }
        else
        {
            currentFrameCounter++;
        }

        opticalFlow.calcOpticalFlowIteration(grayFrame, oldGray, frame, mask, colors);

        // Put in img frame and mask to draw the new flow
        cv::Mat img;
        add(frame, mask, img);

        // Draw on a new image the rectangle updated
        cv::Mat newInitialFrame = firstFrame.clone();
        cv::Rect box = cv::boundingRect(verifiedFirstFrameKP);
        // Clamping at the borders, intersection between box and the rectangle representing the whole image
        // to avoid getting a box out of the image
        box &= cv::Rect(0, 0, img.cols, img.rows);
        cv::rectangle(newInitialFrame, box, cv::Scalar(0, 0, 255), 2);

        imshow("Frame" + std::to_string(i), img);
        cv::imshow("Initial Frame", newInitialFrame);
        int keyboard = cv::waitKey();
        if (keyboard == 'q' || keyboard == 27)
            break;

        oldGray = grayFrame.clone();

        std::cout << "Keypoints size: " << survivingKP.size() << std::endl;
    }
}
template void processImageSequence<cv::SIFT>(std::vector<cv::Mat> images, cv::Ptr<cv::SIFT> detector, float minMotion);
template bool extractFeatures<cv::SIFT>(const cv::Ptr<cv::SIFT> &detector, const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);