/*
@author: Sirio Trentin
*/
#include "evaluate_performace.hpp"
#include "utils.hpp"
#include "process_images.hpp"
#include <iostream>
#include <fstream>

static double computeIoU(const cv::Rect &a, const cv::Rect &b) {
    cv::Rect intersezione = a & b;
    double interArea = intersezione.area();
    double unionArea = a.area() + b.area() - interArea;
    if (unionArea <= 0.0) return 0.0;
    return interArea / unionArea;
}

EvaluationResults evaluatePerformance(const std::string& directory_progetto) {
    const std::filesystem::path dataPath = std::filesystem::path(directory_progetto) / "resources" / "data";
    const std::filesystem::path labelsPath = std::filesystem::path(directory_progetto) / "resources" / "labels";
    const std::filesystem::path reportImagesPath = std::filesystem::path(directory_progetto) / "report" / "images";

    if (!std::filesystem::exists(reportImagesPath)) {
        std::filesystem::create_directories(reportImagesPath);
    }

    // ORB per la valutazione
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    constexpr float minMotion = 0.2f;

    EvaluationResults results;
    results.correct_folders = 0;
    results.total_folders = 0;

    for (auto &entry : std::filesystem::directory_iterator(dataPath)) {
        std::string folderName = entry.path().filename().string();
        if (folderName == ".DS_Store" || !std::filesystem::is_directory(entry.path())) continue;
        std::cout << "Folder evaluation: " << folderName << std::endl;

        // caricamento immagini
        std::vector<cv::Mat> images;
        loadImages(entry.path().string(), images);

        std::string currentImagePath = (reportImagesPath / (folderName + ".png")).string();

        // codice di verifica sulla cartella corrente
        cv::Rect predicted;
        processImageSequence(images, detector, minMotion, &predicted, currentImagePath);

        // lettura labels, con squirrel serve specificare se double o single.
        std::filesystem::path labelFile;
        if(folderName == "squirrel") {
            labelFile = labelsPath / folderName / "double_squirrel" / "0000.txt";
        }
        else{
            labelFile = labelsPath / folderName / "0000.txt";
        }

        // creazione rettangolo corretto
        std::ifstream ifs(labelFile);
        int x1,y1,x2,y2;
        ifs >> x1 >> y1 >> x2 >> y2;
        cv::Rect gt(x1,y1, x2 - x1, y2 - y1);

        double iou = computeIoU(predicted, gt);
        std::cout << "  -IoU = " << iou << std::endl;
        results.ious.push_back(iou);
        results.categories.push_back(folderName);
        results.imagePaths.push_back(currentImagePath);
        if (iou > 0.5) results.correct_folders++;
        results.total_folders++;

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
