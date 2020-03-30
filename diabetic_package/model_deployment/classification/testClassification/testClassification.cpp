#include "testClassification.h"

int main() {
    // 计时变量
    clock_t start, finish, startAllTime, endAllTime;
    double costTime = 0, allTime = 0, allTimeMultiThread = 0;

    // 数据获取
    string testPath = "..\\data";
    string filepath = testPath + "/*.bmp";
    vector<std::string> files;
    cv::glob(filepath, files, false);
    int imgNum = files.size();

    // 预测相关变量
    string configPath = "../../model/classification";
    bool useMultiThreading = false;
    Classification testPb = Classification(configPath,
        useMultiThreading, // 是否使用多线程
        { "softmax","classes" });
    int batchSize = 3;// 单批处理图片数量

    startAllTime = clock();
    for (int i = 0; i < imgNum / batchSize; i++) {
        // mat vector生成
        vector<cv::Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            // 需求：输入图片是彩色图像
            imgs.push_back(cv::imread(files[i*batchSize + b], 0));
        }

        // 图像预测
        start = clock();
        vector<cv::Mat> res;
        if (useMultiThreading) {
            thread t(&Classification::predict, &testPb, imgs, &res);
            t.join();
        }
        else {
            testPb.predict(imgs, &res);
        }
        finish = clock();

        // 结果输出
        vector<int> classes = testPb.cvMat2Int(res[0]);
        vector<vector<float>> softmax = testPb.cvMat2Float(res[1]);

        for (int b = 0; b < batchSize; b++) {
            cout << "本批次第" << b << "张图类别：" << classes[b] << endl;
            cout << "本批次第" << b << "张图的概率值：" << softmax[b][0]
                << " " << softmax[b][1] << endl;
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "第" << i + 1 << "个batch" << "耗时" << costTime << endl;
    }

    endAllTime = clock();
    double meanTime = (endAllTime - startAllTime) / CLOCKS_PER_SEC / double(imgNum / batchSize);
    double meanPicTime = allTime / (double)imgNum;

    std::cout << "单batch流程平均时间:" << meanTime << std::endl;
    std::cout << "单张图片预测平均时间：" << meanPicTime << endl;

    system("pause");
    return 0;
}