#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Sequential.h"
#include "Parallel.cuh"



int main(){
  processImageParallelCombined("../imgs/lena_rgb.png", "../execution_times.csv");
  return 0;
}
