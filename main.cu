#include "Sequential.h"
#include "Parallel.cuh"
int main() {
    // Update the paths as needed
    string path = "../imgs/lena_4k.jpg";
    string csvPath = "../execution_times.csv";

    // processImage(path, csvPath);
    processImageParallel(path, csvPath);

    return 0;
}