#pragma once
#include <iostream>
#include <cmath>
#include <thread>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

extern "C" {
    int process_frame(unsigned char *arr, size_t height, size_t width, size_t depth);
}
