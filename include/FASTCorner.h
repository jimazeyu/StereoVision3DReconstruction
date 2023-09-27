// common
#include <iostream>
#include <cstdlib>
#include <cmath>

// stl
#include <vector>
#include <utility>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>

// namespaces
using namespace std;
using namespace cv;
using namespace Eigen;

// 超参数(hyperparameter)定义
const int Pixel_Gap_Threshold = 50;

// 常量定义
// 关键圆环，上下左右，距离3格
const int KeyCircleThreshold = 3; // 对于KeyCircle上的点，至少要满足3个
const int KeyCircle[4][2] =
    {{-3, 0},
     {0, -3},
     {0, 3},
     {3, 0}};

// 周围圆环去除关键圆环剩余的圆环
const int RestCircleThreshold = 9; // 对于RestCircle上的点，至少要满足3个
const int RestCircle[12][2] =
    {{-3, -1}, {-3, 1}, {-2, -2}, {-2, 2}, {-1, -3}, {-1, 3}, {1, -3}, {1, 3}, {2, -2}, {2, 2}, {3, -1}, {3, 1}};

const int Directions[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

// 手写非极大值抑制（不包括二值化）
void myNonMaxSuppress(const Mat &src, Mat &dst)
{
    const int Rows = src.rows;
    const int Cols = src.cols;

    Mat dstCopy = Mat::zeros(src.size(), CV_8UC1); // 临时保存一份，否则后面会出问题
    // 遍历每一个像素，判断是否为极大值。是则保留，不是则删去。
    for (int row = 0; row < Rows; row++) // 遍历行
    {
        for (int col = 0; col < Cols; col++) // 遍历列
        {
            unsigned char currentValue = src.at<unsigned char>(row, col); // 当前点的值
            // cout<<row<<" "<<col<<endl;
            vector<unsigned char> surroundValues; // 当前点周围的值
            for (int dir = 0; dir < 8; dir++)
            {
                int surroundRow = row + Directions[dir][0];
                int surroundCol = col + Directions[dir][1];
                // 对于超出范围的进行周围点进行跳过
                if (surroundRow < 0 || surroundCol < 0 || surroundRow > (Rows - 1) || surroundCol > (Cols - 1))
                    continue;
                
                surroundValues.push_back(src.at<unsigned char>(surroundRow, surroundCol));
            }
            // cout<<int(src.at<unsigned char>(row, col))<<endl;
            unsigned char surroundMaxValue = *max_element(surroundValues.begin(), surroundValues.end());
            if (currentValue > surroundMaxValue) // 是局部极大值
                dstCopy.at<unsigned char>(row, col) = (unsigned char)(255);
        }
    }
    // cout<<dstCopy<<endl;
    dst = dstCopy.clone();
}

//
void myFastCornerDetection(const Mat &srcImg, Mat &dstMat, const int gapThreshold = Pixel_Gap_Threshold)
{
    if (srcImg.channels() != 1)
    {
        cout << "[myFastCornerDetection] :: srcImg has too much channels." << endl;
        exit(-1);
    }
    const int Rows = srcImg.rows;
    const int Cols = srcImg.cols;
    dstMat = Mat::zeros(srcImg.size(), CV_8UC1);

    // 遍历每一个像素
    for (int row = 0; row < Rows; row++)
    {
        for (int col = 0; col < Cols; col++)
        {
            int keyCircleCounter = 0;                                               // keyCircle点计数器
            int restCircleCounter = 0;                                              // restCircle点计数器
            const int centerPointValue = (int)(srcImg.at<unsigned char>(row, col)); // 当前点的值
            int circlePointValue = 0;                                               // 圆上点的值

            // 先对keyCircle进行判断
            for (int circlePointId = 0; circlePointId < 4; circlePointId++)
            {
                int surroundRow = row + KeyCircle[circlePointId][0];
                int surroundCol = col + KeyCircle[circlePointId][1];
                // 判断圆上的点是否在范围内
                if (surroundRow < 0 || surroundCol < 0 || surroundRow > (Rows - 1) || surroundCol > (Cols - 1))
                    continue;
                circlePointValue = (int)(srcImg.at<unsigned char>(surroundRow, surroundCol));
                if (circlePointValue < centerPointValue - gapThreshold || circlePointValue > centerPointValue + gapThreshold)
                {
                    ++keyCircleCounter;
                }
            }
            if (keyCircleCounter < 3) // 对于关键圆上的点不能满足3个及以上就删除
                continue;

            // 在对restCircle进行判断
            for (int circlePointId = 0; circlePointId < 12; circlePointId++)
            {
                int surroundRow = row + RestCircle[circlePointId][0];
                int surroundCol = col + RestCircle[circlePointId][1];
                // 判断圆上的点是否在范围内
                if (surroundRow < 0 || surroundCol < 0 || surroundRow > (Rows - 1) || surroundCol > (Cols - 1))
                    continue;
                circlePointValue = (int)(srcImg.at<unsigned char>(surroundRow, surroundCol));
                if (circlePointValue < centerPointValue - gapThreshold || circlePointValue > centerPointValue + gapThreshold)
                {
                    ++restCircleCounter;
                }
            }
            if (restCircleCounter < 9) // 对于剩余圆上的点不能满足9个及以上就删除
                continue;

            // 通过上述检查的点进行记录
            dstMat.at<unsigned char>(row, col) = srcImg.at<unsigned char>(row, col);
            // 这里不能直接赋值255，否则后面非极大值抑制会出错
            // 可能出现：阈值低，一堆点挨在一起，都没有被认为是极大值；而阈值高，这些点被拆开，都被认为是极大值点。
        }
    }
    myNonMaxSuppress(dstMat, dstMat);
    // cout<<dstMat<<endl;
}

void myFastManualThresholdTrackBarCallback(int threshold, void *srcImgPtr)
{
    const Mat srcImg = *(Mat *)(srcImgPtr);
    Mat dstMat;

    myFastCornerDetection(srcImg, dstMat, threshold);
    imshow("ManualGetThreshold", dstMat);
}

int myFastCornerDetectionManualThreshold(const Mat &srcImg, Mat &dstMat)
{
    if (srcImg.channels() != 1)
    {
        cout << "[myFastCornerDetection] :: srcImg has too much channels." << endl;
        exit(-1);
    }

    namedWindow("ManualGetThreshold", WINDOW_AUTOSIZE);
    imshow("ManualGetThreshold", srcImg);

    int threshold = 3;
    int thresholdMax = 255;

    createTrackbar("FAST Threshold", "ManualGetThreshold", &threshold,
                   thresholdMax, myFastManualThresholdTrackBarCallback, (void *)&srcImg);

    waitKey(0);
    destroyAllWindows();

    myFastCornerDetection(srcImg, dstMat, threshold);

    cout << "[myFASTCornerDetection] :: Manual choice threshold : [" << threshold << "]." << endl;
    return threshold;
}

// 将自己实现的二值化的Mat转为vector形式，方便后续特征匹配
void myConvertFASTMatToPairVector(const Mat &thresholdMat, vector<pair<int, int>> &cornerPoints, bool showPoints=false)
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
                cornerPoints.push_back(pair<int, int>(row, col));
            }
        }
    }
    cornerPoints.shrink_to_fit();
}

// 将自己实现的二值化的Mat转为vector形式，方便后续特征匹配
void myConvertFASTMatToCVKeyPointVector(const Mat &thresholdMat, vector<KeyPoint> &cornerPoints, bool showPoints=false)
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