#include "testSegmentationClassification.h"
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;



int main() {
    // 时间变量
    clock_t start, finish;
    double costTime = 0, allTime = 0;

    // 图片数据读取
    string testPath = "../../data/";
    string filePath = testPath + "*.png";
    vector<string> files;
    glob(filePath, files, false);
    int imgNum = files.size();
    string path = "../results/";//结果保存路径

    // 分割变量
    string segmentationConfigPath = "../../model/segmentation/";
    bool useMultiThreading = false;


    // 分类变量
    string classificationConfigPath = "../../model/classification";
 
    segmentationClassification segmentationClassificationModel = segmentationClassification(
        classificationConfigPath,
        segmentationConfigPath,
        useMultiThreading
    );


    // 结果预测
    int batchSize = 1;
    for (int i = 0; i < imgNum / batchSize; ++i) {
        // 分割输入图像向量生成
        vector<Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            if ((i*batchSize + b) > imgNum) {
                // 越界判断
                break;
            }
            // 分割要求灰度图像输入
            imgs.push_back(imread(files[i*batchSize + b], 0));
        }

        // 图像预测
        start = clock();
        vector<Mat> results;
        results = segmentationClassificationModel.segmentationClassificationPredict(imgs, batchSize);
      
        finish = clock();

        // 结果保存
        for (int b = 0; b < batchSize; ++b) {
            string name = files[i*batchSize + b].substr(strlen(testPath.c_str()));
            cout << "文件名" << name << endl;
            imwrite(path + name, results[b]);
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "第" << i << "个batch花费的时间" << costTime << endl;
    }
    double meanTime = allTime / double(imgNum);
    double meanTimePerBatch = allTime / double(imgNum / batchSize);
    cout << "单张图片预测平均耗时：" << meanTime << endl;
    cout << "batch为" << batchSize << "时，每个batch预测平均耗时" << meanTimePerBatch << endl;
    
    system("pause");
    return 0;
}
