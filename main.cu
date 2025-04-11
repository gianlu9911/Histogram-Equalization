#include "Parallel.cuh"
#include "Sequential.h"
#include <opencv2/core/utils/logger.hpp>

int main(){
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    processImage("../imgs/lena_rgb.png", "../execution_times.csv");
    processImageCuda("../imgs/lena_rgb.png");
}