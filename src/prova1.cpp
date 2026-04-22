#include <opencv2/opencv.hpp>
#include <filesystem>

void opticalFLow(std::vector<std::string> images) {
    // TODO: tunare per ogni immagine e in base al detector utilizzato, ognuna ha la sua
    constexpr float minMotion = 0.1f;

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

    cv::Mat frame = cv::imread(images[0]);
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

    std::cout << "Processing: " << images[0] << std::endl;

    std::vector<cv::Point2f> currKP;


    // Detect keypoints using SIFT
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    // TODO: impostare la mask in base ad ogni immagine
    // TODO: tunare bene per tutte le immagini (in particolare bird e squirrel)
    detector->detect(frame, keypoints);
    cv::KeyPoint::convert(keypoints, currKP);

    std::cout << "Keypoints size: " << currKP.size() << std::endl;


    /*
    // Detect features using goodFeaturesToTrack
    // Take first frame and find corners in it
    // TODO: impostare la mask in base ad ogni immagine
    // TODO: tunare bene per tutte le immagini (in particolare bird e squirrel)
    goodFeaturesToTrack(grayFrame, currKP, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);

    std::cout << "Keypoints size: " << currKP.size() << std::endl;
    */
    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    std::vector<cv::Point2f> newKP;

    // Read and process all images
    for (size_t i = 1; i < images.size(); i++) {
        cv::Mat oldFrame = frame.clone();
        cv::Mat oldGray = grayFrame.clone();
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (frame.empty()) {
            std::cerr << "Could not read image: " << images[i] << std::endl;
            continue;
        }

        std::cout << "Processing: " << images[i] << std::endl;

        // Optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(oldGray, grayFrame, currKP, newKP, status, err, cv::Size(15,15), 2, criteria);

        std::vector<cv::Point2f> moving_pts;

        for (uint j = 0; j < currKP.size(); j++) {
            // Select good points, status == 1 if flow has been found
            if (status[j] == 1) {   // TODO: keep track of every keypoint to compare the various vectors of keypoints and find the object in the first image
                float d = cv::norm(newKP[j] - currKP[j]);  // Euclidean distance

                if (d > minMotion) {
                    moving_pts.push_back(newKP[j]);
                    // draw the tracks
                    cv::line(mask,newKP[j], currKP[j], colors[j], 2);
                    circle(frame, newKP[j], 5, colors[j], -1);
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
        currKP = moving_pts;

        std::cout << "Keypoints size: " << currKP.size() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "You have to specify the folder" << argv[0];
        return -1;
    }

    std::string folderName = argv[1];

    std::string folderPath = "../../resources/data/" + folderName;   // your folder containing frames
    std::vector<std::string> images;

    // Read all image files from directory
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        images.push_back(entry.path().string());
    }

    // Sort file names so frames are processed in order
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        std::cerr << "No images found in folder: " << folderPath << std::endl;
        return -1;
    }

    opticalFLow(images);

    return 0;
}