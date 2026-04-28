/*
@author: Sirio Trentin
*/
#include "evaluate_performace.hpp"
#include "utils.hpp"
#include "process_images.hpp"
#include <iostream>
#include <fstream>

static double computeIoU(const cv::Rect &a, const cv::Rect &b) {
    cv::Rect intersection = a & b;
    double interArea = intersection.area();
    double unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0.0) return 0.0;
    return interArea / unionArea;
}

void saveBoundingBox(const std::string& folderName, const cv::Rect& predicted, std::ofstream& ofs) {
    int px1 = predicted.x;
    int py1 = predicted.y;
    int px2 = predicted.x + predicted.width;
    int py2 = predicted.y + predicted.height;

    std::cout << "  -Predicted BB (" << folderName << "): " << px1 << " " << py1 << " " << px2 << " " << py2 << std::endl;
    if (ofs.is_open()) {
        ofs << folderName << " " << px1 << " " << py1 << " " << px2 << " " << py2 << "\n";
    }
}

EvaluationResults evaluatePerformance(const std::string& directory_progetto) {
    const std::filesystem::path dataPath = std::filesystem::path(directory_progetto) / "resources" / "data";
    const std::filesystem::path labelsPath = std::filesystem::path(directory_progetto) / "resources" / "labels";
    const std::filesystem::path reportImagesPath = std::filesystem::path(directory_progetto) / "report" / "images";

    if (!std::filesystem::exists(reportImagesPath)) {
        std::filesystem::create_directories(reportImagesPath);
    }

    // SIFT for evaluation
    cv::Ptr<cv::Feature2D> detector = cv::SIFT::create();
    constexpr float minMotion = 0.2f;

    EvaluationResults results;
    results.correct_folders = 0;
    results.total_folders = 0;

    const std::filesystem::path outBoxesFile = std::filesystem::path(directory_progetto) / "predicted_bounding_boxes.txt";
    std::ofstream ofs(outBoxesFile);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << outBoxesFile << " for writing bounding boxes." << std::endl;
    }

    for (auto &entry : std::filesystem::directory_iterator(dataPath)) {
        std::string folderName = entry.path().filename().string();
        if (folderName == ".DS_Store" || !std::filesystem::is_directory(entry.path())) continue;
        std::cout << "Folder evaluation: " << folderName << std::endl;

        // load images
        std::vector<cv::Mat> images;
        loadImages(entry.path().string(), images);

        std::string currentImagePath = (reportImagesPath / (folderName + ".png")).string();

        // run evaluation on current folder
        cv::Rect predicted;
        processImageSequence(images, detector, minMotion, &predicted, currentImagePath);

        // read labels, for squirrel we need to specify either double or single
        std::filesystem::path labelFile;
        if(folderName == "squirrel") {
            labelFile = labelsPath / folderName / "double_squirrel" / "0000.txt";
        }
        else{
            labelFile = labelsPath / folderName / "0000.txt";
        }

        // create ground truth rectangle
        std::ifstream ifs(labelFile);
        int gt_x1, gt_y1, gt_x2, gt_y2;
        ifs >> gt_x1 >> gt_y1 >> gt_x2 >> gt_y2;
        cv::Rect gt(gt_x1, gt_y1, gt_x2 - gt_x1, gt_y2 - gt_y1);

        saveBoundingBox(folderName, predicted, ofs);

        double iou = computeIoU(predicted, gt);
        std::cout << "  -IoU = " << iou << std::endl;
        results.ious.push_back(iou);
        results.categories.push_back(folderName);
        results.imagePaths.push_back(currentImagePath);
        if (iou > 0.5) results.correct_folders++;
        results.total_folders++;

    }

    if (ofs.is_open()) {
        ofs.close();
    }

    results.meanIoU = 0.0;
    for (double v : results.ious) results.meanIoU += v;
    results.meanIoU = results.meanIoU / static_cast<double>(results.ious.size());

    results.accuracy = static_cast<double>(results.correct_folders) / results.total_folders;

    std::cout << "Results:" << std::endl;
    std::cout << "  -Evaluated Folders: " << results.total_folders << std::endl;
    std::cout << "  -Mean IoU: " << results.meanIoU << std::endl;
    std::cout << "  -Accuracy: " << results.accuracy << std::endl;
    
    // Show the saved images with bounding boxes
    for (const auto& path : results.imagePaths) {
        cv::Mat resultImg = cv::imread(path);
        if (!resultImg.empty()) {
            cv::imshow("Evaluation Result: " + path, resultImg);
        }
    }
    if (!results.imagePaths.empty()) {
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    return results;
}
