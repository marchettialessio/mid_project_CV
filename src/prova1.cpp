#include <opencv2/opencv.hpp>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "You have to specify the folder" << argv[0];
        return -1;
    }

    std::string folderName = argv[1];

    std::string folderPath = "../../resources/data/" + folderName;   // your folder containing frames
    std::vector<std::string> imagePaths;

    // Read all image files from directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        imagePaths.push_back(entry.path().string());
    }

    // Sort file names so frames are processed in order
    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        std::cerr << "No images found in folder: " << folderPath << std::endl;
        return -1;
    }

    const float minMotion = 0.1f;
    // Create some random colors to draw the moving features
    std::vector<cv::Scalar> colors;
    cv::RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r,g,b));
    }

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;

    cv::Mat frame = cv::imread(imagePaths[0]);
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

    std::cout << "Processing: " << imagePaths[0] << std::endl;

    // Features detection with SIFT
    /*
    detector->detect(grayFrame, keypoints);
    cv::Mat output;
    cv::drawKeypoints(frame, keypoints, output);
    */
    //cv::imshow("Features0", output);

    std::vector<cv::Point2f> p0, p1, firstFeatures;
    // Take first frame and find corners in it
    goodFeaturesToTrack(grayFrame, p0, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    firstFeatures = p0;
    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());

    // Read and process all images
    for (size_t i = 1; i < imagePaths.size(); i++) {
        cv::Mat oldFrame = frame.clone();
        cv::Mat oldGray = grayFrame.clone();
        frame = cv::imread(imagePaths[i]);
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (frame.empty()) {
            std::cerr << "Could not read image: " << imagePaths[i] << std::endl;
            continue;
        }

        std::cout << "Processing: " << imagePaths[i] << std::endl;

        // Features detection with sift
        /*
        detector->detect(grayFrame, keypoints);
        cv::drawKeypoints(frame, keypoints, output);
        */
        //cv::imshow("Features" + std::to_string(i), output);

        // Optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(oldGray, grayFrame, p0, p1, status, err, cv::Size(15,15), 2, criteria);

        std::vector<cv::Point2f> good_new, moving_pts;

        for (uint j = 0; j < p0.size(); j++) {
            // Select good points, status == 1 if flow has been found
            if (status[j] == 1) {
                good_new.push_back(p1[j]);

                float d = cv::norm(p1[j] - p0[j]);  // Euclidean distance

                if (d > minMotion) {
                    moving_pts.push_back(p1[j]);
                    // draw the tracks
                    cv::line(mask,p1[j], p0[j], colors[j], 2);
                    circle(frame, p1[j], 5, colors[j], -1);
                }
            }
        }
        cv::Mat img;
        add(frame, mask, img);

        cv::Rect box = cv::boundingRect(moving_pts);
        // Clamping at the borders
        box &= cv::Rect(0, 0, img.cols, img.rows);
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);

        imshow("Frame" + std::to_string(i), img);
        int keyboard = cv::waitKey();
        if (keyboard == 'q' || keyboard == 27)
            break;

        // Update the previous frame and previous points
        p0 = good_new;
    }

    return 0;
}