#include "Parallel.cuh"
#include "Sequential.h"

int main(){
    processImage("../imgs/lena_rgb.png", "../execution_times.csv");
    processImageCuda("../imgs/lena_rgb.png", "../execution_times.csv");
    return 0;
}

