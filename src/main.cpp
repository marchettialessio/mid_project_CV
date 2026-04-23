#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "utils.hpp"
#include "process_images.hpp"
#include <iostream>
#include <vector>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "You have to specify the folder" << argv[0];
        return -1;
    }

    // Get the foldetpath
    std::string folderName = argv[1];

    const std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) / "resources" / "data" / "frog" / folderName;
    std::vector<cv::Mat> images;

    // I load all images
    loadImages(path.string(), images);

    if (images.empty())
    {
        std::cerr << "Error: no valid images found." << std::endl;
        return -1;
    }

    constexpr float minMotion = 0.005f;
    // bird
    // car 0.4f SIFT
    // frog 0.005f SIFT
    // sheep 0.05
    // squirrel

    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

    processImageSequence<cv::SIFT>(images, detector, minMotion);

    while (cv::waitKey() != 'q' && cv::waitKey() != 27) {
        // Wait until user presses 'q' or 'ESC'
    }

    return 0;
}