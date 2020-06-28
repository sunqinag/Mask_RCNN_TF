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

class __declspec(dllexport) Classification {
public:
    Classification(std::string configPath,
        bool useMultithread,
        std::vector<std::string> outputName
    );
    ~Classification();
    int predict(std::vector<cv::Mat> imgs,
        std::vector<cv::Mat>* output
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
    std::vector<int> cvMat2Int(cv::Mat classes);
    std::vector<std::vector<float>> cvMat2Float(cv::Mat softmax);
private:
    int loadModel(std::string modelPath);
    // Mat ת tensor
    void cvMatTotfTensor(std::vector<cv::Mat>& input,
        tensorflow::Tensor & outputTensor
    );
    void tfTensorToMat(tensorflow::Tensor& inputTensor,
        cv::Mat& output
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
