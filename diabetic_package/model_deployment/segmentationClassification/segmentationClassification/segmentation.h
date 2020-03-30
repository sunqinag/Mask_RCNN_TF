#pragma once

#include <stdio.h>
#include <vector>
#include <thread>
#include <mutex>
#include <fstream>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"


class __declspec(dllexport) Segmentation {
public:
    Segmentation(std::string configPath,
        bool useMultithread,
        std::vector<std::string> outputName
    );
    ~Segmentation();
    int predict(std::vector<cv::Mat> imgs,
        std::vector<std::vector<cv::Mat>>* output
    );
    cv::Mat ucharTocvMat(uchar* src,
        int height,
        int width,
        int channel
    );
    cv::Mat ucharTocvMat(uchar* src,
        cv::Size size,
        int channel
    );
private:
    int loadModel(std::string model_path);
    void cvMatTotfTensor(std::vector<cv::Mat>& input,
        tensorflow::Tensor & outputTensor,
        std::vector<cv::Size>& imgSizes
    );
    void tfTensorTocvMat(tensorflow::Tensor& inputTensor,
        std::vector<cv::Mat>& output,
        std::vector<cv::Size>& imgSizes
    );
    void checkParam();

private:
    int inputHeight;
    int inputWidth;
    int inputChannel;
    bool useMultithreading;
    std::vector<std::string> outputLayerName;
    tensorflow::Session* pSession;
};