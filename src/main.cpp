/*
@author: Sirio Trentin
*/
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "utils.hpp"
#include "process_images.hpp"
#include "evaluate_performace.hpp"
#include "report_generator.hpp"
#include <iostream>
#include <vector>
#include <filesystem>

int main(int argc, char** argv) {

    //if nothing specified, run evaluation and generate report
    if (argc == 1) {
        EvaluationResults results = evaluatePerformance(PROJECT_SOURCE_DIR);
        generateLatexResults(results, std::string(PROJECT_SOURCE_DIR) + "/report");
        return 0;
    }

    // otherwise, run the tracking on the specified folder with the specified detector
    std::string folderName = argv[1];

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

    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();

    // Call processImageSequence with savePath="display" to show results
    processImageSequence(images, detector, minMotion, nullptr, "display");

    while (cv::waitKey() != 'q' && cv::waitKey() != 27) {
        // Wait until user presses 'q' or 'ESC'
    }

    return 0;
}