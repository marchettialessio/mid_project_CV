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

void evaluatePerformance(const std::string& directory_progetto) {
    const std::filesystem::path dataPath = std::filesystem::path(directory_progetto) / "resources" / "data";
    const std::filesystem::path labelsPath = std::filesystem::path(directory_progetto) / "resources" / "labels";

    // ORB per la valutazione
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    constexpr float minMotion = 0.2f;

    std::vector<double> ious;
    int correct_folders = 0;
    int total_folders = 0;

    for (auto &entry : std::filesystem::directory_iterator(dataPath)) {
        std::string folderName = entry.path().filename().string();
        std::cout << "Valutazione cartella: " << folderName << std::endl;

        // caricamento immagini
        std::vector<cv::Mat> images;
        loadImages(entry.path().string(), images);

        // codice di verifica sulla cartella corrente
        cv::Rect predicted;
        processImageSequence(images, detector, minMotion, &predicted, false);

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
        std::cout << "  IoU = " << iou << std::endl;
        ious.push_back(iou);
        if (iou > 0.5) correct_folders++;
        total_folders++;

    }

    double meanIoU = 0.0;
    for (double v : ious) meanIoU += v;
    meanIoU = meanIoU/static_cast<double>(ious.size());

    double accuracy = correct_folders;

    std::cout << "Risultati:" << std::endl;
    std::cout << " Cartelle valutate: " << total_folders << std::endl;
    std::cout << "  IoU media: " << meanIoU << std::endl;
    std::cout << "  Accuracy: " << accuracy << std::endl;
}
