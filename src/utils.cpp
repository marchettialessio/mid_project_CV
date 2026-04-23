#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void loadImages(const std::string &directory, std::vector<cv::Mat> &images)
{
    try
    {
        std::error_code ec;

        if (!std::filesystem::exists(directory, ec))
        {
            if (ec)
            {
                std::cerr << "Error: unable to access directory '" << directory
                          << "' (" << ec.message() << ")." << std::endl;
            }
            else
            {
                std::cerr << "Error: the directory does not exist." << std::endl;
            }
            return;
        }

        std::vector<std::filesystem::path> image_paths;

        //Iterate through the directory and collect valid image paths
        for (std::filesystem::directory_iterator it(directory, ec), end; it != end; it.increment(ec))
        {
            if (ec)
            {
                std::cerr << "Error: cannot iterate directory '" << directory
                          << "' (" << ec.message() << ")." << std::endl;
                if (ec == std::errc::operation_not_permitted || ec == std::errc::permission_denied)
                {
                    std::cerr << "Hint (macOS): grant VS Code access to Desktop in"
                              << " System Settings > Privacy & Security > Files and Folders."
                              << std::endl;
                }
                return;
            }

            const auto &entry = *it;
            if (entry.is_regular_file() && cv::haveImageReader(entry.path().string()))
            {
                image_paths.push_back(entry.path());
            }
        }

        //Sort the image paths to ensure they are in the correct order
        std::sort(image_paths.begin(), image_paths.end(),
                  [](const std::filesystem::path &a, const std::filesystem::path &b)
                  {
                      return a.filename().string() < b.filename().string();
                  });

        for (const auto &image_path : image_paths)
        {
            cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);

            if (!img.empty())
            {
                images.push_back(img);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

std::vector<cv::KeyPoint> filterKeypointsAboveThreshold(const std::vector<cv::KeyPoint> &keypoints, const std::vector<double> &distances, int threshold)
{
    std::vector<cv::KeyPoint> filteredKeypoints;
    filteredKeypoints.reserve(keypoints.size());

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        if (distances[i] > static_cast<double>(threshold))
        {
            filteredKeypoints.push_back(keypoints[i]);
        }
    }

    return filteredKeypoints;
}