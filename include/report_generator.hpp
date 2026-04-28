/*
@author: Alessio Marchetti
*/
#ifndef REPORT_GENERATOR_HPP
#define REPORT_GENERATOR_HPP

#include "evaluate_performace.hpp"
#include <string>

void generateLatexResults(const EvaluationResults& results, const std::string& output_dir);

#endif
