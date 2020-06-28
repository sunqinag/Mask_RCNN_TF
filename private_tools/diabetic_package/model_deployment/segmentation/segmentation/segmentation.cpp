#include "Segmentation.h"

// 临界资源管理锁
std::mutex session_source;

// public

Segmentation::Segmentation(std::string configPath, 
    bool useMultithread, 
    std::vector<std::string> outputName = { "classes" }) {
    std::string modelPath = "";


    // 输入路径判断最后有没有加//,没有则自动填充
    char last = configPath[configPath.length() - 1];
    //std::string
    if (last != '/') {
        std::string doubleSlash = configPath.substr(configPath.length() - 1, 2);
        std::string dSlash = "\\";
        if (!(doubleSlash == dSlash)) {
            configPath.insert(configPath.length(), "/");
        }
    }

    // 解析配置文件
    char buffer[100];
    std::ifstream configFile(configPath + "model_config.txt");
    if (!configFile.is_open()) {
        std::cout << "请检查配置文件是否存在！" << std::endl;
    }

    while (!configFile.eof()) {
        configFile.getline(buffer, 100);
        std::string keyName = std::strtok(buffer, ":");
        std::string ValueName = std::strtok(NULL, ":");
        if (keyName == "model_name") {
            modelPath = configPath + ValueName;
        }
        else if (keyName == "input_height") {
            inputHeight = std::atoi(ValueName.c_str());
        }
        else if (keyName == "input_width") {
            inputWidth = std::atoi(ValueName.c_str());
        }
        else if (keyName == "input_channel") {
            inputChannel = std::atoi(ValueName.c_str());
        }
    }
    checkParam();

    if (outputName.size() > 1) {
        outputLayerName = { "classes","softmax" };
    }
    else {
        outputLayerName = outputName;
    }
    useMultithreading = useMultithread;
    loadModel(modelPath);
}

Segmentation::~Segmentation() {
    if (pSession != nullptr) {
        pSession->Close();
        pSession = nullptr;
    }
}

int Segmentation::predict(std::vector<cv::Mat> imgs, std::vector<std::vector<cv::Mat>>* output) {
    int batchSize = imgs.size();
    std::vector<cv::Size> imgSizes;

    // 生成输入tensor
    tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT,
        tensorflow::TensorShape({ batchSize,inputHeight,inputWidth,inputChannel }));

    cvMatTotfTensor(imgs, inputTensor, imgSizes);

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>("input_img", inputTensor));

    // 获取并解析输出tensor
    std::vector<tensorflow::Tensor> outputs;
    if (useMultithreading) {
        session_source.lock();
    }
    tensorflow::Status status = pSession->Run(inputs, outputLayerName, {}, &outputs);
    if (useMultithreading) {
        session_source.unlock();
    }
    if (!status.ok()) {
        std::cout << "前向传播预测失败" << std::endl;
        return -1;
    }

    // 获得预测结果 
    for (int i = 0; i < outputLayerName.size(); i++) {
        std::vector<cv::Mat> labelImg;
        tfTensorTocvMat(outputs[i], labelImg, imgSizes);
        output->push_back(labelImg);
    }
    return 0;
}

cv::Mat Segmentation::ucharTocvMat(uchar * src, int height, int width, int channel) {
    return cv::Mat(cv::Size(width, height), CV_32FC(channel), src);
}

cv::Mat Segmentation::ucharTocvMat(uchar* src, cv::Size size, int channel) {
    return cv::Mat(size, CV_32FC(channel), src);
}

// private

// 参数检查
void Segmentation::checkParam() {
    if (inputHeight <= 0) {
        std::cout << "请检查配置文件中是否设定了参数：输入图片高" << std::endl;
    }
    if (inputWidth <= 0) {
        std::cout << "请检查配置文件中是否设定了参数：输入图片宽" << std::endl;
    }
    if (inputChannel <= 0) {
        std::cout << "请检查配置文件中是否设定了参数：输入图片通道数" << std::endl;
    }
}

// 模型加载
int Segmentation::loadModel(std::string modelPath) {
    tensorflow::GraphDef graph;

    // 模型加载
    tensorflow::Status ret = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelPath, &graph);
    if (!ret.ok()) {
        std::cout << "模型加载失败！" << std::endl;
        std::cout << "模型路径：" << modelPath << std::endl;
        return -1;
    }

    // session配置
    auto options = tensorflow::SessionOptions();
    NewSession(options, &pSession);

    // 模型图与session相关联
    ret = pSession->Create(graph);
    if (!ret.ok()) {
        std::cout << "模型与session关联失败,请检查session创建是否成功！" << std::endl;
        return -1;
    }
    return 0;
}

// Mat 转 tensor
void Segmentation::cvMatTotfTensor(std::vector<cv::Mat>& input, tensorflow::Tensor & outputTensor, std::vector<cv::Size>& imgSizes) {
    // 4:(b,h,w,c)
    int batchSize = input.size();
    int depth = inputChannel;
    float * p = outputTensor.flat<float>().data();

    for (int b = 0; b < batchSize; b++) {
        // 记录图片原始尺寸
        imgSizes.push_back(input[b].size());

        // mat转tensor
        cv::resize(input[b], input[b], cv::Size(inputWidth, inputHeight));
        cv::Mat tempMat(cv::Size(inputWidth, inputHeight), CV_32FC(depth), p);
        input[b].convertTo(tempMat, CV_32FC(depth));
        p += inputWidth*inputHeight*depth;
    }

}

// tensor 转 mat
void Segmentation::tfTensorTocvMat(tensorflow::Tensor& inputTensor, std::vector<cv::Mat>& output, std::vector<cv::Size>& imgSizes) {
    tensorflow::TensorShape inputTensorShape = inputTensor.shape();

    int batch = inputTensorShape.dim_size(0);
    int height = inputTensorShape.dim_size(1);
    int width = inputTensorShape.dim_size(2);
    int depth;
    if (inputTensorShape.dims() < 4) {
        // (b,h,w):三维tensor转换
        depth = 1;
        uchar* p = inputTensor.flat<uchar>().data();

        for (int b = 0; b < batch; ++b) {
            cv::Mat singleImg8 = cv::Mat(height, width, CV_8UC(depth), p);
            cv::resize(singleImg8, singleImg8, imgSizes[b], 0, 0, cv::INTER_NEAREST);
            output.push_back(singleImg8);
            p += height*width*depth;
        }
    }
    else {
        // (b,h,w,c):四维tensor转换
        depth = inputTensorShape.dim_size(3);
        float* p = inputTensor.flat<float>().data();

        for (int b = 0; b < batch; ++b) {
            cv::Mat singleImg = cv::Mat(cv::Size(height, width), CV_32FC(depth), p);
            cv::resize(singleImg, singleImg, imgSizes[b], 0, 0, cv::INTER_LINEAR);
            output.push_back(singleImg);
            p += height*width*depth;
        }
    }
}