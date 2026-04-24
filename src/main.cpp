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
    int selection = std::stoi(argv[2]);
    const std::filesystem::path path = std::filesystem::path(PROJECT_SOURCE_DIR) / "resources" / "data" /  folderName;
    std::vector<cv::Mat> images;

    // I load all images
    loadImages(path.string(), images);

    if (images.empty())
    {
        std::cerr << "Error: no valid images found." << std::endl;
        return -1;
    }

    constexpr float minMotion = 0.2f;
    // bird
    // car 0.4f SIFT
    // frog 0.005f SIFT
    // sheep 0.05
    // squirrel
    cv::Ptr<cv::Feature2D> detector;
    if(selection==0){
        detector = cv::SIFT::create();
    }
    else if(selection==1){
        detector = cv::ORB::create();
    }
    else if(selection==2){
        detector = cv::KAZE::create();
    }
    else{
        std::cerr << "Invalid selection. Use 0 for SIFT and 1 for ORB 2 for KAZE" << std::endl;
        return -1;
    }

    processImageSequence<cv::Feature2D>(images, detector, minMotion);

    while (cv::waitKey() != 'q' && cv::waitKey() != 27) {
        // Wait until user presses 'q' or 'ESC'
    }

    return 0;
}