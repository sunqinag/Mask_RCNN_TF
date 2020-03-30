#include "testSegmentation.h"
#include "../../classification/classification/classification.h"
#include <opencv2/imgproc/types_c.h>

void testSegmentation();

int main() {
    testSegmentation();
    return 0;
}

void testSegmentation() {
    // 计时变量
    clock_t start, finish, startAllTime, endAllTime;
    double costTime, allTime, allTimeMultiThread = 0;

    // 数据获取
    string testPath = "../../data";
    string filepath = testPath + "/*.png";
    vector<std::string> files;
    cv::glob(filepath, files, false);
    int imgNum = files.size();

    // 预测相关变量
    string configPath = "../../model/segmentation";
    bool useMultiThreading = false;
    Segmentation testPb = Segmentation(configPath,
        useMultiThreading,//是否使用多线程
        { "softmax","classes" });
    int batchSize = 2;// 单批处理图片数量

    startAllTime = clock();
    for (int i = 0; i < imgNum / batchSize; i++) {
        // mat vector生成
        vector<cv::Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            imgs.push_back(cv::imread(files[i*batchSize + b], 0));
        }

        // 图像预测
        start = clock();
        vector<vector<cv::Mat>> res;
        if (useMultiThreading) {
            thread t(&Segmentation::predict, &testPb, imgs, &res);
            t.join();
        }
        else {
            testPb.predict(imgs, &res);
        }
        finish = clock();

        // 图片写入
        string path = "../results/";
        for (int b = 0; b < batchSize; b++) {
            string name = files[i * batchSize + b].substr(strlen(testPath.c_str()) + 1);
            cout << name << endl;
            cv::Mat temp = res[0][b];
            cv::imwrite(path + name, temp);//res[0][b]
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "第" << i + 1 << "个batch" << "耗时" << costTime << endl;
    }
    endAllTime = clock();
    double meanTime = (endAllTime - startAllTime) / CLOCKS_PER_SEC / double(imgNum);
    double meanPicTime = allTime / (double)imgNum;

    std::cout << "单线程平均时间:" << meanTime << std::endl;
    std::cout << "单张图片预测平均时间：" << meanPicTime << endl;

    system("pause");
}