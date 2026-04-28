/*
@author: Alessio Marchetti
*/
#include "report_generator.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

void generateLatexResults(const EvaluationResults& results, const std::string& output_dir) {
    std::filesystem::path dir(output_dir);
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
    
    std::filesystem::path filepath = dir / "_results.tex";
    std::ofstream ofs(filepath);
    
    if (!ofs.is_open()) {
        std::cerr << "Failed to open " << filepath << " for writing." << std::endl;
        return;
    }
    
    // Begin a single table region placing both side-by-side using minipages 
    // to bring them closer
    ofs << "\\begin{table}[htbp]\n"
        << "\\centering\n"
        << "\\begin{minipage}{0.45\\textwidth}\n"
        << "\\centering\n"
        << "\\begin{tabular}{|l|c|}\n"
        << "\\hline\n"
        << "\\textbf{Category} & \\textbf{IoU} \\\\\n"
        << "\\hline\n";
    
    for (size_t i = 0; i < results.categories.size(); ++i) {
        ofs << results.categories[i] << " & " 
            << std::fixed << std::setprecision(4) << results.ious[i] << " \\\\\n";
    }
    
    ofs << "\\hline\n"
        << "\\end{tabular}\n"
        << "\\caption{IoU per category.}\n"
        << "\\label{tab:iou_per_category}\n"
        << "\\end{minipage} \\hfill\n"
        << "\\begin{minipage}{0.45\\textwidth}\n"
        << "\\centering\n"
        << "\\begin{tabular}{|l|c|}\n"
        << "\\hline\n"
        << "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        << "\\hline\n"
        << "Mean IoU & " << std::fixed << std::setprecision(4) << results.meanIoU << " \\\\\n"
        << "Accuracy & " << std::fixed << std::setprecision(4) << results.accuracy << " (" << results.correct_folders << "/" << results.total_folders << ")" << " \\\\\n"
        << "\\hline\n"
        << "\\end{tabular}\n"
        << "\\caption{Overall Performance Metrics.}\n"
        << "\\label{tab:overall_metrics}\n"
        << "\\end{minipage}\n"
        << "\\end{table}\n\n"
        << "\\clearpage\n"
        << "\\section*{Initial Frame Tracking Results}\n"
        << "\\begin{figure}[htbp]\n"
        << "\\centering\n";

    for (size_t i = 0; i < results.categories.size(); ++i) {
        if (!results.imagePaths[i].empty()) {
            // Using relative path for the latex document since images in report/images and main.tex in report/
            std::filesystem::path imgPath(results.imagePaths[i]);
            std::string relativePath = "images/" + imgPath.filename().string();
            
            ofs << "\\begin{minipage}{0.32\\textwidth}\n"
                << "\\centering\n"
                << "\\includegraphics[width=\\textwidth]{" << relativePath << "}\n"
                << "\\vspace{0.2cm}\n"
                << "\\textbf{" << results.categories[i] << "}\n"
                << "\\end{minipage}\\hfill\n";
        }
    }
    
    ofs << "\\caption{Initial Frame for each category with tracked bounding box.}\n"
        << "\\end{figure}\n\n";
    
    ofs.close();
}
