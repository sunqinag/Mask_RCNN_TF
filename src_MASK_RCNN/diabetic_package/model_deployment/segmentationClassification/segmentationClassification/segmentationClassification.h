#pragma once
#include "classification.h"
#include "segmentation.h"
#include <opencv2/imgproc/types_c.h>

class __declspec(dllexport) segmentationClassification {
public:
    segmentationClassification(std::string classPath,
        std::string segmentPath,
        bool multiThread);
    std::vector<cv::Mat> segmentationClassificationPredict(
        std::vector<cv::Mat>& imgs,
        int batchSize
    );
private:
    std::vector<cv::Point> getRectLUCornerRDCorner(cv::Rect boundRectBox, cv::Mat& img);
private:
    std::string classificationPath;
    std::string segmentationPath;
    bool useMultiThreading;
    Classification classificationModel;
    Segmentation segmentationModel;
};