#include "process_images.hpp"
#include "utils.hpp"

void calcOpticalFlowPipeline(std::vector<cv::Point2f> initialKP, std::vector<std::string> images, float minMotion) {

}

template<typename T>
void processImageSequence(std::vector<cv::Mat> images, cv::Ptr<T> detector, float minMotion) {

    std::vector<cv::Scalar> colors = generateRandomColors(100);    // Generate random colors for drawing

    // Read initial frame and get grayscale
    cv::Mat firstFrame = images[0];
    cv::Mat firstFrameGray;
    cv::cvtColor(firstFrame, firstFrameGray, cv::COLOR_BGR2GRAY);

    // Detect keypoints using parameter detector
    std::vector<cv::Point2f> firstFrameKP;
    cv::KeyPoint::convert(detector->detect(firstFrameGray, keypoints), firstFrameKP);    // Convert to point in 2D

    std::cout << "Keypoints size: " << firstFrameKP.size() << std::endl;

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(firstFrame.size(), firstFrame.type());
    // Set frame and keypoints for the next iteration
    cv::Mat oldGray = firstFrameGray;
    std::vector<cv::Point2f> currKP = firstFrameKP;
    std::vector<cv::Point2f>* usefulKP = &firstFrameKP;    // Pointer to vector of useful keypoints to draw the rectangle

    // TODO: se KP.size va a 0 break

    // Read and process images
    for (size_t i = 1; i < images.size(); i++) {
        // Read frame 'i' and get grayscale
        cv::Mat frame = images[i];
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // Optical flow
        std::vector<cv::Point2f> newKP;
        std::vector<uchar> status;
        std::vector<float> err;
        // TODO: understand criteria
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(oldGray, grayFrame, currKP, newKP, status, err, cv::Size(15,15), 2, criteria);

        std::vector<cv::Point2f> moving_pts, good_new;
        std::vector<cv::Point2f> currInitialKP = *usefulKP;

        // TODO: sistemare erase, trovare altro modo per ottenere un nuovo vettore di keypoints iniziali con possibile dimensione minore
        // Scan the vectors of keypoints
        for (uint j = 0; j < currKP.size(); j++) {
            // Select good points, status == 1 if flow has been found
            if (status[j] == 1) {
                float d = cv::norm(newKP[j] - currKP[j]);  // Euclidean distance
                // TODO: impostare che se una feature si è mossa un tot di volte è definitiva, perché ad esempio nella frog alcune feature finiscono fuori dall'immagine e vengono perse
                if (d >= minMotion) {    // if the point moved more or equal than minMotion -> track the flow
                    moving_pts.push_back(currInitialKP[j]);
                    good_new.push_back(newKP[j]);

                    // Draw the tracks
                    cv::line(mask,newKP[j], currKP[j], colors[j % colors.size()], 2);
                    circle(frame, newKP[j], 5, colors[j % colors.size()], -1);
                }
            }
        }
        // Put in img frame and mask to draw the new flow
        cv::Mat img;
        add(frame, mask, img);

        // Draw on a new image the rectangle updated
        cv::Mat newInitialFrame = firstFrame.clone();
        *usefulKP = moving_pts;
        cv::Rect box = cv::boundingRect(moving_pts);
        // Clamping at the borders, intersection between box and the rectangle representing the whole image
        // to avoid getting a box out of the image
        box &= cv::Rect(0, 0, img.cols, img.rows);
        cv::rectangle(newInitialFrame, box, cv::Scalar(0, 0, 255), 2);

        imshow("Frame" + std::to_string(i), img);
        cv::imshow("Initial Frame", newInitialFrame);
        int keyboard = cv::waitKey();
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Update the old frame and current points
        currKP = good_new;
        oldGray = grayFrame.clone();

        std::cout << "Keypoints size: " << currKP.size() << std::endl;

        std::cout << "Moving points size: " << currKP.size() << std::endl;
    }
}