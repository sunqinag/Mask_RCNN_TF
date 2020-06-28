#include "classification.h"

// 临界资源管理锁
std::mutex session_source_class;

// public

Classification::Classification(std::string configPath,
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

    //解析配置文件
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

Classification::~Classification() {
    if (pSession != nullptr) {
        pSession->Close();
        pSession = nullptr;
    }
}

int Classification::predict(std::vector<cv::Mat> imgs, std::vector<cv::Mat>* output) {
    int batchSize = imgs.size();

    // 生成输入tensor
    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
        tensorflow::TensorShape({ batchSize,inputHeight,inputWidth,inputChannel }));
    cvMatTotfTensor(imgs, input_tensor);

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>("img", input_tensor));

    // 获取并解析输出tensor
    std::vector<tensorflow::Tensor> outputs;
    if (useMultithreading) {
        session_source_class.lock();
    }
    tensorflow::Status status = pSession->Run(inputs, outputLayerName, {}, &outputs);
    if (useMultithreading) {
        session_source_class.unlock();
    }
    if (!status.ok()) {
        std::cout << "前向传播预测失败" << std::endl;
        return -1;
    }

    // 获取返回值
    for (int layerOrder = 0; layerOrder < outputLayerName.size(); layerOrder++) {
        cv::Mat results;
        tfTensorToMat(outputs[layerOrder], results);
        output->push_back(results);
    }
    return 0;
}

cv::Mat Classification::ucharTocvMat(uchar * src, int height, int width, int channel) {
    return cv::Mat(cv::Size(width, height), CV_32FC(channel), src);
}

cv::Mat Classification::ucharTocvMat(uchar* src, cv::Size size, int channel) {
    return cv::Mat(size, CV_32FC(channel), src);
}

std::vector<int> Classification::cvMat2Int(cv::Mat classes) {
    // TODO: 输入参数检查
    return (std::vector<int>)(classes.reshape(1, 1));
}

std::vector<std::vector<float>> Classification::cvMat2Float(cv::Mat softmax) {
    // 每行为单张图片的概率数组
    std::vector<std::vector<float>> results;
    float * p = (float *)softmax.data;
    for (int batch = 0; batch < softmax.rows; ++batch) {
        std::vector<float> singleResults(&p[0], &p[softmax.cols]);
        results.push_back(singleResults);
        p += softmax.cols;
    }
    return results;
}

// private

// 参数检查
void Classification::checkParam() {
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
int Classification::loadModel(std::string modelPath) {
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
void Classification::cvMatTotfTensor(std::vector<cv::Mat>& input,
    tensorflow::Tensor & outputTensor) {
    // 4:(b,h,w,c)
    int batchSize = input.size();
    int depth = inputChannel;

    float * p = outputTensor.flat<float>().data();

    for (int b = 0; b < batchSize; b++) {
        // 如果输入图像为单通道图像
        if (input[b].channels() < 3) {
            cvtColor(input[b], input[b], cv::COLOR_GRAY2BGR);
        }

        // mat转tensor
        cv::resize(input[b], input[b], cv::Size(inputWidth, inputHeight));
        cv::Mat tempMat(cv::Size(inputWidth, inputHeight), CV_32FC(depth), p);
        input[b].convertTo(tempMat, CV_32FC(depth));
        p += inputWidth*inputHeight*depth;
    }
}

// tensor 转 float向量(概率结果)
void Classification::tfTensorToMat(tensorflow::Tensor& inputTensor, cv::Mat& output) {
    tensorflow::TensorShape inputTensorShape = inputTensor.shape();

    int batch = inputTensorShape.dim_size(0);
    if (inputTensorShape.dims() < 2) {
        // (b):classes结果
        long long* p = inputTensor.flat<INT64>().data();

        cv::Mat classes(batch, 1, CV_32SC1);
        int* classesPtr = (int *)classes.data;
        for (int b = 0; b < batch; b++) {
            *classesPtr = (int)*p;
            p += 1;
            classesPtr += 1;
        }
        output = classes.clone();
        //output.push_back(classes);
    }
    else {
        // (b,c):softmax结果
        int probabilityLength = inputTensorShape.dim_size(1);
        float* p = inputTensor.flat<float>().data();

        cv::Mat softmax(batch, probabilityLength, CV_32FC1, p);
        output = softmax.clone();
    }
}
