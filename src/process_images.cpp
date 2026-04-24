#include "process_images.hpp"
#include "utils.hpp"
#include <opencv2/features2d.hpp>
#include "optical_flow.hpp"

template<typename T>
void processImageSequence(std::vector<cv::Mat> images, cv::Ptr<T> detector, float minMotion) {

    std::vector<cv::Scalar> colors = generateRandomColors(100);    // Generate random colors for drawing

    // Read initial frame and get grayscale
    cv::Mat firstFrame = images[0];
    cv::Mat firstFrameGray;
    cv::cvtColor(firstFrame, firstFrameGray, cv::COLOR_BGR2GRAY);

    // Detect keypoints using parameter detector
    std::vector<cv::Point2f> firstFrameKP;
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(firstFrameGray, keypoints);
    cv::KeyPoint::convert(keypoints, firstFrameKP);    // Convert to point in 2D

    std::cout << "Keypoints size: " << firstFrameKP.size() << std::endl;

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(firstFrame.size(), firstFrame.type());
    // Set frame and keypoints for the next iteration
    cv::Mat oldGray = firstFrameGray;
    std::vector<cv::Point2f> currKP = firstFrameKP;

    //this is the outut of optical flow => i want to know keypoints of first frame that survive during iterations
    // To keep track of the initial position of the keypoints for drawing the rectangle
    std::vector<cv::Point2f> survivingKP = firstFrameKP;    

    // TODO: se KP.size va a 0 break (done by the second to last condition in the loop).
    OpticalFlow opticalFlow(currKP, survivingKP, minMotion);
    size_t prev_surviving_count = survivingKP.size();
    // Read and process images
    for (size_t i = 1; i < images.size(); i++) {
        // Read frame 'i' and get grayscale
        cv::Mat frame = images[i];
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        opticalFlow.calcOpticalFlowIteration(grayFrame, oldGray, frame, mask, colors);

        // Put in img frame and mask to draw the new flow
        cv::Mat img;
        add(frame, mask, img);

        // Draw on a new image the rectangle updated
        cv::Mat newInitialFrame = firstFrame.clone();
        survivingKP = opticalFlow.getSurvivingKP();
        cv::Rect box = cv::boundingRect(survivingKP);

        // Clamping at the borders, intersection between box and the rectangle representing the whole image
        // to avoid getting a box out of the image
        box &= cv::Rect(0, 0, img.cols, img.rows);
        cv::rectangle(newInitialFrame, box, cv::Scalar(0, 0, 255), 2);
        std::cout << "Surviving keypoints: " << survivingKP.size() << std::endl;
        std::cout << "prev_surviving_count: " << prev_surviving_count << std::endl;

        //If the number of keypoints is much less than the previous iteration, we change the threshold.
        if (prev_surviving_count >= survivingKP.size()*3 && survivingKP.size() < 40)
        {
            minMotion = std::max(minMotion - 0.01f, 1e-6f); //Update of minMotion.

            // Reinitialize optical flow and keypoints to the first frame
            opticalFlow = OpticalFlow(firstFrameKP, firstFrameKP, minMotion);
            currKP = firstFrameKP;
            survivingKP = firstFrameKP;
            oldGray = firstFrameGray.clone();
            mask = cv::Mat::zeros(firstFrame.size(), firstFrame.type());

            // Reset loop to start processing from the second frame
            i = 0; 
            prev_surviving_count = survivingKP.size();
            continue;
        }

        else if(survivingKP.size() <= 30) {
            std::cout << "Not enough keypoints to track. Ending process." << std::endl;
            cv::imshow("Frame" + std::to_string(i), img);
            cv::imshow("Initial Frame", newInitialFrame);
            int keyboard = cv::waitKey();
            if (keyboard == 'q' || keyboard == 27)
                break;
        }
        else if(i==images.size()-1) {
            cv::imshow("Frame" + std::to_string(i), img);
            cv::imshow("Initial Frame", newInitialFrame);
            int keyboard = cv::waitKey();
            if (keyboard == 'q' || keyboard == 27)
                break;
        }
     
        oldGray = grayFrame.clone();
        prev_surviving_count = survivingKP.size();
        std::cout << "Keypoints size: " << survivingKP.size() << std::endl;
    }
}

template void processImageSequence<cv::Feature2D>(std::vector<cv::Mat> images, cv::Ptr<cv::Feature2D> detector, float minMotion);