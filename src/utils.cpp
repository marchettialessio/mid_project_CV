#include "utils.hpp"
#include <iostream>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Function to load all images from a specified directory
void loadImages(const std::string &directory, std::vector<cv::Mat> &images)
{
    try
    {
        if (!std::filesystem::exists(directory))
        {
            std::cerr << "Error: the directory does not exist." << std::endl;
            return;
        }

        std::vector<std::filesystem::path> image_paths;

        //Iterate through the directory and collect valid image paths
        for (const auto &entry : std::filesystem::directory_iterator(directory))
        {
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