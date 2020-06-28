#include "segmentationClassification.h"

segmentationClassification::segmentationClassification(std::string classPath, 
    std::string segmentPath, 
    bool multiThread) :
    useMultiThreading(multiThread),
    classificationModel(classPath, useMultiThreading, { "classes" }), 
    segmentationModel(segmentPath,useMultiThreading,{ "classes" }) {
}

std::vector<cv::Mat> segmentationClassification::segmentationClassificationPredict(std::vector<cv::Mat>& imgs, int batchSize){
    using namespace std;
    using namespace cv;

    // 分割预测
    vector<vector<Mat>> segmentationResults;
    if (useMultiThreading) {
        thread t(&Segmentation::predict, &segmentationModel, imgs, &segmentationResults);
        t.join();
    }
    else {
        segmentationModel.predict(imgs, &segmentationResults);
    }
    // 分类:单图处理    
    for (int imgIndex = 0; imgIndex < batchSize; imgIndex++) {
        Mat segmentationLabel = segmentationResults[0][imgIndex];

        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        int imgHeight = segmentationLabel.rows;
        int imgWidth = segmentationLabel.cols;

        findContours(segmentationLabel, contours, hierarchy,
            CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));        

        // 分类要求彩色图像输入
        Mat imgRGB;
        cvtColor(imgs[imgIndex], imgRGB, CV_GRAY2RGB);

        // 分类patch处理
        // 分类输入图片向量生成
        vector<Mat> subImgs;
        vector<Rect> boundRect(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            boundRect[i] = boundingRect(Mat(contours[i]));
            vector<Point> angulars = getRectLUCornerRDCorner(boundRect[i], segmentationLabel);
            Point start = angulars[0];
            Point end = angulars[1];
            Mat subImg(imgRGB, Rect(start, end));
            subImgs.push_back(subImg);
        }

        // 预测
        if (subImgs.size() >= 1) {
            vector<Mat> resClass;
            classificationModel.predict(subImgs, &resClass);
            vector<int> classes;
            classes = classificationModel.cvMat2Int(resClass[0]);

            // 结果回填给标签图片
            for (int i = 0; i < classes.size(); ++i) {
                if (classes[i] == 0) {
                    Mat subLabel(segmentationLabel, Rect(boundRect[i].tl(), boundRect[i].br()));
                    subLabel = Scalar::all(0);
                }
            }
        }
    }
    return segmentationResults[0];
}

std::vector<cv::Point> segmentationClassification::getRectLUCornerRDCorner(cv::Rect boundRectBox,cv::Mat& img){
    int rectHeight = boundRectBox.height;
    int rectWidth = boundRectBox.width;
    std::vector<cv::Point> rectLUCornerRDCorner(2);
    rectLUCornerRDCorner[0].x = boundRectBox.tl().x - 0.5*rectWidth;
    rectLUCornerRDCorner[0].y = boundRectBox.tl().y - 0.5*rectHeight;
    rectLUCornerRDCorner[1].x = boundRectBox.br().x + 0.5*rectWidth;
    rectLUCornerRDCorner[1].y = boundRectBox.br().y + 0.5*rectHeight;

    // 边界值判断
    if (rectLUCornerRDCorner[0].x < 0) rectLUCornerRDCorner[0].x = 0;
    if (rectLUCornerRDCorner[0].y < 0) rectLUCornerRDCorner[0].y = 0;
    if (rectLUCornerRDCorner[1].x > img.cols) rectLUCornerRDCorner[1].x = img.cols;
    if (rectLUCornerRDCorner[1].y > img.rows) rectLUCornerRDCorner[1].y = img.rows;
    return rectLUCornerRDCorner;
}

