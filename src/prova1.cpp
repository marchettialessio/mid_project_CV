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

    // Read initial frame and get grayscale
    cv::Mat initialFrame = cv::imread(images[0]);
    cv::Mat initialGray;
    cv::cvtColor(initialFrame, initialGray, cv::COLOR_BGR2GRAY);

    std::cout << "Processing: " << images[0] << std::endl;

    // Detect keypoints using SIFT
    std::vector<cv::Point2f> initialKP;
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    // TODO: impostare la mask in base ad ogni immagine
    // TODO: tunare bene per tutte le immagini (in particolare bird e squirrel)
    detector->detect(initialGray, keypoints);
    cv::KeyPoint::convert(keypoints, initialKP);    // Convert to point in 2D

    /*
    // Detect features using goodFeaturesToTrack
    // Take first frame and find corners in it
    // TODO: impostare la mask in base ad ogni immagine
    // TODO: tunare bene per tutte le immagini (in particolare bird e squirrel)
    goodFeaturesToTrack(grayFrame, currKP, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    */

    std::cout << "Keypoints size: " << initialKP.size() << std::endl;

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(initialFrame.size(), initialFrame.type());
    cv::Mat oldGray = initialGray;
    std::vector<cv::Point2f> currKP = initialKP;

    // TODO: se KP.size va a 0 break

    // Read and process all images
    for (size_t i = 1; i < images.size(); i++) {
        // Read frame i and get grayscale
        cv::Mat frame = cv::imread(images[i]);
        cv::Mat grayFrame;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        if (frame.empty()) {
            std::cerr << "Could not read image: " << images[i] << std::endl;
            continue;
        }

        std::cout << "Processing: " << images[i] << std::endl;

        // Optical flow
        std::vector<cv::Point2f> newKP;
        std::vector<uchar> status;
        std::vector<float> err;
        // TODO: understand criteria
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
        calcOpticalFlowPyrLK(oldGray, grayFrame, currKP, newKP, status, err, cv::Size(15,15), 2, criteria);

        std::vector<cv::Point2f> moving_pts, good_new;

        // TODO: sistemare erase, trovare altro modo per ottenere un nuovo vettore di keypoints iniziali con possibile dimensione minore
        // Scan the vectors of keypoints
        for (uint j = currKP.size()-1; j > 0; j--) { // TODO: salta l'iterazione 0
            // Select good points, status == 1 if flow has been found
            if (status[j] == 1) {
                float d = cv::norm(newKP[j] - currKP[j]);  // Euclidean distance

                if (d >= minMotion) {    // if the point moved more or equal than minMotion -> track the flow
                    moving_pts.push_back(initialKP[j]);
                    good_new.push_back(newKP[j]);

                    // Draw the tracks
                    cv::line(mask,newKP[j], currKP[j], colors[j], 2);
                    circle(frame, newKP[j], 5, colors[j], -1);
                } else {
                    initialKP.erase(initialKP.begin() + j);
                }
            } else {
                initialKP.erase(initialKP.begin() + j);
            }
        }
        // Put in img frame and mask to draw the new flow
        cv::Mat img;
        add(frame, mask, img);

        // Draw on a new image the rectangle updated
        cv::Mat newInitialFrame = initialFrame.clone();
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "You have to specify the folder" << argv[0];
        return -1;
    }

    // Get the foldetpath
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

    while (cv::waitKey() != 'q' && cv::waitKey() != 27) {
        // Wait until user presses 'q' or 'ESC'
    }

    return 0;
}