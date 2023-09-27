// common
#include <iostream>
#include <cstdlib>
#include <cmath>

// stl
#include <vector>
#include <utility>

// eigen
# include <eigen3/Eigen/Core>
# include <eigen3/Eigen/Dense>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>

// namespaces
using namespace std;
using namespace cv;
using namespace Eigen;

// 超参数(hyperparameter)定义
const float MY_CORNER_RESPONSE_ALPHA = 0.05f;
const float HARRIS_CORNER_RESPONSE_THRESHOLD = 125.0f;

// 常量(const)定义
const int MY_BORDER_CONSTANT = 0;
const int MY_NOUSE_WINDOW_TYPE = 0;
const int MY_GAUSSIAN_WINDOW_TYPE = 1;
const int MY_BOX_WINDOW_TYPE = 2;


Mat PrewittOperatorX = (Mat_<float>(3,3)<<-1, 0, 1,
                                          -1, 0, 1,
                                          -1, 0, 1);
Mat PrewittOperatorY = (Mat_<float>(3,3)<<-1,-1,-1,
                                           0, 0, 0,
                                           1, 1, 1);

Mat SobelOperatorX = (Mat_<float>(3,3)<<-1, 0, 1,
                                        -2, 0, 2,
                                        -1, 0, 1);
Mat SobelOperatorY = (Mat_<float>(3,3)<<-1,-2,-1,
                                         0, 0, 0,
                                         1, 2, 1);

const int Directions[8][2] =   {{-1,-1},{-1,0},{-1,1},
                                { 0,-1},       { 0,1},
                                { 1,-1},{ 1,0},{ 1,1}};



// 自己实现OpenCV中的copyMakeBorder函数，目前只支持四边扩展一样大，用常数补齐
void myCopyMakeBorder(const Mat& src,Mat& dst, int borderType=MY_BORDER_CONSTANT, \
                      int borderSize=1, int borderValue=0)
{
    int withBorderHeight = src.rows+2*borderSize;
    int withBorderWidth = src.cols+2*borderSize;
    Size withBorderSize = Size(withBorderWidth, withBorderHeight);

    float affineTX =static_cast<float>(borderSize), affineTY = static_cast<float>(borderSize);
    float warpAffineValues[6] = {1.0f,0.0f,affineTX,0.0f,1.0f,affineTY};
    Mat affineTransformer = Mat(2,3,CV_32F,warpAffineValues);

    if(src.channels()==1)
    {
        cout<<"[myCopyMakeBorder] :: src has 1 channel."<<endl;
        warpAffine(src,dst,affineTransformer,withBorderSize,BORDER_CONSTANT,0);
        Mat mask = Mat::ones(withBorderSize, CV_8UC1);  // 构建模板，将src放入中间
        // 模板中心块改为1，矩形填充
        rectangle(mask,Rect(borderSize,borderSize,src.cols,src.rows),Scalar(0),-1);
        multiply(mask,Scalar(borderValue),mask);  // 模板乘以相应倍数
        add(dst,mask,dst);
        imshow("mask",mask);
        imshow("src",src);
        imshow("dst",dst);
        waitKey(0);
        destroyAllWindows();
    }
    else
    {
        cout<<"[myCopyMakeBorder] :: src has too much channels, which is not supportable."<<endl;
        destroyAllWindows();
        exit(-1);
    }
    return;
}

// 自己实现的高斯模糊
void myGaussianBlur(Mat& src,Mat& dst)
{

}

// 自己实现的获取窗函数
void myGetWindowFunctionKernel(Mat& kernel, const int windowSize=3, \
                               const int windowType=MY_BOX_WINDOW_TYPE)
{
    if(windowType==MY_GAUSSIAN_WINDOW_TYPE)
    {
        // hyperparameter
        Mat gaussianX = getGaussianKernel(windowSize, 1, CV_32F);
        Mat gaussianY;
        transpose(gaussianX,gaussianY);
        kernel = gaussianX*gaussianY;
    }
    else if(windowType==MY_BOX_WINDOW_TYPE)
    {
        kernel = Mat::ones(Size(windowSize,windowSize),CV_32F);
        multiply(kernel,Scalar(1.0f/pow(windowSize,2)),kernel);
    }
    else if(windowType==MY_NOUSE_WINDOW_TYPE)
    {
        int windowCenter = windowSize/2;
        kernel = Mat::zeros(Size(windowSize,windowSize),CV_32F);
        kernel.at<float>(windowCenter,windowCenter) = 1.0f;
    }
    // cout<<kernel<<endl;
}

// 测试myGetWindowFunctionKernel函数是否正常工作
void myTestWindowKernel()
{
    Mat kernel;
    myGetWindowFunctionKernel(kernel,3,MY_NOUSE_WINDOW_TYPE);
    myGetWindowFunctionKernel(kernel,3,MY_BOX_WINDOW_TYPE);
    myGetWindowFunctionKernel(kernel,3,MY_GAUSSIAN_WINDOW_TYPE);
}

// 给定2x2矩阵并计算其response function
float myComputeCornerResponse(const Mat& mat2x2,const float alpha=MY_CORNER_RESPONSE_ALPHA)
{
    float cornerResponse = 0.0f;
    if(mat2x2.size()!=Size(2,2))
    {
        cout<<"[myGet2By2MatEigenValues] :: wrong mat size."<<endl;
        exit(-1);
    }

    float eigenSum=0.0f, eigenProduct=0.0f;
    eigenSum = mat2x2.at<float>(0,0)+mat2x2.at<float>(1,1);
    eigenProduct = mat2x2.at<float>(0,0)*mat2x2.at<float>(1,1)-mat2x2.at<float>(0,1)*mat2x2.at<float>(1,0);
    // cout<<mat2x2.at<float>(0,0)<<"  "<<mat2x2.at<float>(0,1)<<"  "<<mat2x2.at<float>(1,0)<<"  "<<mat2x2.at<float>(1,1)<<endl;
    // cout<<"eigenSum         : "<<eigenSum<<endl;
    // cout<<"eigenProduct     : "<<eigenProduct<<endl;
    
    cornerResponse = eigenProduct - alpha*pow(eigenSum,2);
    // cout<<"cornerResponse   : "<<cornerResponse<<endl;
    return cornerResponse;
}

void myComputeAllCornerResponse(Mat& cornerResponseMat, const vector<vector<Mat>>& IMats)
{
    const int Rows = IMats.size();
    const int Cols = IMats[0].size();
    cornerResponseMat = Mat::zeros(Rows,Cols,CV_32F);
    for(int row=0;row<Rows;row++)
    {
        for(int col=0;col<Cols;col++)
        {
            float tempCornerResponse = myComputeCornerResponse(IMats[row][col]);
            cornerResponseMat.at<float>(row,col) = tempCornerResponse;
        }
    }
}

void myComputeAllMomentMatrix(const Mat& srcImg, vector<vector<Mat>>& IMats,const int windowSize=3, \
                              const int windowType=MY_NOUSE_WINDOW_TYPE)
{
    // 计算Ix,Iy,IxIx,IyIy,IxIy(通道都是 CV_8UC1 类型)
    Mat Ix,Iy,IxIx,IyIy,IxIy;
    filter2D(srcImg,Ix,CV_32F,SobelOperatorX);
    filter2D(srcImg,Iy,CV_32F,SobelOperatorY);
    multiply(Ix,Ix,IxIx);
    multiply(Ix,Iy,IxIy);
    multiply(Iy,Iy,IyIy);
    // cout<<IyIy(Rect(144,50,6,6))<<endl<<IxIy(Rect(144,50,6,6))<<endl<<IxIx(Rect(144,50,6,6))<<endl;

    Mat windowFunctionKernel;
    myGetWindowFunctionKernel(windowFunctionKernel, windowSize, windowType);
    Mat weightIxIx,weightIxIy,weightIyIy;
    filter2D(IxIx,weightIxIx,CV_32F,windowFunctionKernel);
    filter2D(IxIy,weightIxIy,CV_32F,windowFunctionKernel);
    filter2D(IyIy,weightIyIy,CV_32F,windowFunctionKernel);
    // imshow("weightIxIx",weightIxIx);
    // imshow("weightIxIy",weightIxIy);
    // imshow("weightIyIy",weightIyIy);

    // vector<vector<Mat>>IMats;
    for(int row=0;row<srcImg.rows;row++)
    {
        vector<Mat>tempMatVector;
        for(int col=0;col<srcImg.cols;col++)
        {
            Mat tempMat=(Mat_<float>(2,2)<<weightIxIx.at<float>(row,col),weightIxIy.at<float>(row,col),
                                           weightIxIy.at<float>(row,col),weightIyIy.at<float>(row,col));
            tempMatVector.push_back(tempMat);
        }
        IMats.push_back(tempMatVector);
    }
}

// 手写非极大值抑制（包括二值化）
void myNonMaxSuppressWithThreshold(const Mat& src, Mat& dst, \
                                   const float threshold=HARRIS_CORNER_RESPONSE_THRESHOLD)
{
    const int Rows = src.rows;
    const int Cols = src.cols;
    Mat dstCopy = Mat::zeros(src.size(),CV_8UC1);
    // 遍历每一个像素，判断是否为极大值。是则保留，不是则删去。
    for(int row=0;row<Rows;row++)  // 遍历行
    {
        for(int col=0;col<Cols;col++)  // 遍历列
        {
            float currentValue = src.at<float>(row,col);  // 当前点的值
            if(currentValue<threshold)  // 当前点的值低于阈值则直接跳过
                continue;
            // cout<<"("<<row<<","<<col<<"): "<<currentValue<<endl;
            vector<float>surroundValues;  // 当前点周围的值
            for(int dir=0;dir<8;dir++)
            {
                int surroundRow = row+Directions[dir][0];
                int surroundCol = col+Directions[dir][1];
                // 对于超出范围的进行周围点进行跳过
                if(surroundRow<0||surroundCol<0||surroundRow>(Rows-1)||surroundCol>(Cols-1))
                    continue;
                else
                    surroundValues.push_back(src.at<float>(surroundRow,surroundCol));
            }
            // cout<<"vector"<<surroundValues<<endl;
            float surroundMaxValue = *max_element(surroundValues.begin(),surroundValues.end());
            if(currentValue>surroundMaxValue)  // 是局部极大值
                dstCopy.at<unsigned char>(row,col)=255;
        }   
    }
    // cout<<dstCopy<<endl;
    dst = dstCopy.clone();
}


// 自己写的角点检测(输入和输出都是Mat形式)
void myHarrisCornerDetection(const Mat& srcImg, Mat& dstMat, const int windowSize=9, \
                             const int windowboFunctionType=MY_GAUSSIAN_WINDOW_TYPE, \
                             const float threshold=HARRIS_CORNER_RESPONSE_THRESHOLD)
{
    if(srcImg.channels()!=1)
    {
        cout<<"[myHarrisCornerDetection] :: srcImg has too much channels."<<endl;
        exit(-1);
    }
    vector<vector<Mat>>IMats;
    myComputeAllMomentMatrix(srcImg,IMats,windowSize,MY_GAUSSIAN_WINDOW_TYPE);
    Mat cornerResponse;
    myComputeAllCornerResponse(cornerResponse,IMats);
    Mat cornerResponseNonMaxSuppress;
    myNonMaxSuppressWithThreshold(cornerResponse,cornerResponseNonMaxSuppress,threshold);
    dstMat = cornerResponseNonMaxSuppress.clone();
}

// 自己写的角点检测,通过滑动条的方式得到阈值
struct HarrisParameterStruct
{
    const Mat srcImg;
    const int windowSize;
    const int windowTpye;
};

void myHarrisManualThresholdTrackBarCallback(int threshold, void* harrisStructPtr)
{
    const HarrisParameterStruct tempHarrisStruct= *(HarrisParameterStruct*)harrisStructPtr;
    Mat dstMat;

    myHarrisCornerDetection(tempHarrisStruct.srcImg,dstMat,tempHarrisStruct.windowSize,tempHarrisStruct.windowTpye,threshold);    
    imshow("ManualGetThreshold",dstMat);
}
float myHarrisCornerDetectionManualThreshold(const Mat& srcImg, Mat& dstMat, \
                                              const int windowSize=9, \
                                              const int windowboFunctionType=MY_GAUSSIAN_WINDOW_TYPE)
{
    // 先对srcImg图片格式进行判断
    if(srcImg.channels()!=1)
    {
        cout<<"[myHarrisCornerDetection] :: srcImg has too much channels."<<endl;
        exit(-1);
    }
    // 将全部信息进行汇总
    HarrisParameterStruct tempHarrisStruct = {srcImg, windowSize, windowboFunctionType};
    
    // 创建滑动条
    namedWindow("ManualGetThreshold", WINDOW_AUTOSIZE);  // 创建窗口
    imshow("ManualGetThreshold", srcImg);
    float thresholdFloat = 200.0f;
    int thresholdInt = 200;
    int thresholdIntMax = 100000.0f;

    createTrackbar("Harris Threshold", "ManualGetThreshold",&thresholdInt,thresholdIntMax,\
                   myHarrisManualThresholdTrackBarCallback,(void*)&tempHarrisStruct);
    waitKey(0);
    destroyAllWindows();
    
    thresholdFloat = float(thresholdInt);
    cout<<"thresholdFloat : "<<thresholdFloat<<endl;

    myHarrisCornerDetection(srcImg, dstMat, windowSize, windowboFunctionType, thresholdFloat);

    cout<<"[myHarrisCornerDetection] :: Manual choice threshold : ["<<thresholdFloat<<"]."<<endl;
    return thresholdFloat;
}

// 将自己实现的二值化的Mat转为vector形式，方便后续特征匹配
void myConvertHarrisMatToVector(const Mat& harrisThresholdMat, vector<pair<int,int>>& cornerPoints)
{
    const int Rows = harrisThresholdMat.rows;
    const int Cols = harrisThresholdMat.cols;
    const int vectorMaxSize = Rows * Cols;
    cornerPoints.clear();
    cornerPoints.reserve(vectorMaxSize);
    
    for(int row=0;row<Rows;row++)
    {
        for(int col=0;col<Cols;col++)
        {
            // 注意 at 的数据类型一定要设定正确，否则结果无法预估
            if(harrisThresholdMat.at<unsigned char>(row,col)==255) // 是二值化图选出的角点
            {
                cornerPoints.push_back(pair<int,int>(row,col));
            }
        }
    }
    cornerPoints.shrink_to_fit();
    cout<<"[myConvertHarrisMatToVector] :: Vector's Size ["<<cornerPoints.size()<<"]."<<endl;
}

// 将自己实现的二值化的Mat转为vector形式，方便后续特征匹配
void myConvertHarrisMatToCVKeyPointVector(const Mat &thresholdMat, vector<KeyPoint> &cornerPoints, bool showPoints=false)
{
    const int Rows = thresholdMat.rows;
    const int Cols = thresholdMat.cols;
    const int vectorMaxSize = Rows * Cols;
    cornerPoints.clear();
    cornerPoints.reserve(vectorMaxSize);

    for (int row = 0; row < Rows; row++)
    {
        for (int col = 0; col < Cols; col++)
        {
            // 注意 at 的数据类型一定要设定正确，否则结果无法预估
            if (thresholdMat.at<unsigned char>(row, col) == 255) // 是二值化图选出的角点
            {
                float tempKeyPointValue = (float)255;
                KeyPoint tempKeyPoint(float(col),float(row),tempKeyPointValue);
                cornerPoints.push_back(tempKeyPoint);
            }
        }
    }
    cornerPoints.shrink_to_fit();
}
