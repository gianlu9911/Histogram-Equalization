#include "Sequential.h"

int main() {
    // Update the paths as needed
    string path = "../imgs/lena.jpg";
    string csvPath = "../execution_times.csv";

    processImage(path, csvPath);

    return 0;
}
