#include "Parallel.cuh"
#include "Sequential.h"

using namespace cv;
using namespace std;

int main(){
    processImage("../imgs/lena_rgb.png", "../execution_times.csv");
    processImageCuda("../imgs/lena_rgb.png", "../execution_times.csv");
    return 0;
}

