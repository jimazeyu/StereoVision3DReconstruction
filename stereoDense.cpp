/*
 * @Author: Jimazeyu
 * @Date: 2022-06-02 03:35:51
 * @LastEditors: Jimazeyu
 * @LastEditTime: 2022-06-08 10:08:07
 * @FilePath: /Stereo_Bighomework/stereoDense.cpp
 * @Description: 稠密建图
 * @Version: 1.0
 */
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>
#include "include/utils.hpp"
using namespace std;
using namespace Eigen;
// 相机内参
double fx = 1758.23, fy = 1758.23, cx = 953.34, cy = 953.34;
// 相机基线
double b = 0.11153;

int main(int argc, char **argv)
{
    // 读取图片
    string left_file, right_file;
    left_file = argv[1];
    right_file = argv[2];
    cv::Mat left = cv::imread(left_file, 0);
    cv::Mat right = cv::imread(right_file, 0);
    // 判断是否读取成功
    if (left.empty() || right.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        return -1;
    }
    // 显示图片
    show_two_eyes(left, right);
    // SBGM稠密建图
    cv::Mat img_left = cv::imread(left_file, 0);
    cv::Mat img_right = cv::imread(right_file, 0);
    // 判断是否读取成功
    if (img_left.empty() || img_right.empty())
    {
        cout << "Could not open or find the image!\n"
             << endl;
        return -1;
    }

    // 初始化SBGM参数
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    sgbm->setBlockSize(5);
    sgbm->setP1(8 * 3 * 3);
    sgbm->setP2(32 * 3 * 3);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(160);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
    // 计算拼接图像
    cv::Mat disp_sgbm, disp;
    sgbm->compute(img_left, img_right, disp_sgbm);
    // 将深度图转换为8位图像
    disp_sgbm.convertTo(disp, CV_32F, 1.0 / 16.0f);
    // 显示
    cv::imshow("disp", disp / 96);
    while (true)
    {
        if (cv::waitKey(10) == 27)
        {
            cv::destroyAllWindows();
            break;
        }
    }

    // 点云数组
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
    // 便利像素点，利用disparity和相机内参计算点深度信息，将点存入数组中
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++)
        {
            if (disp.at<float>(v, u) <= 0.0 || disp.at<float>(v, u) >= 96.0)
                continue;
            Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色
            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disp.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;
            pointcloud.push_back(point);
        }
    // 画出点云
    showPointCloud(pointcloud);
    return 0;
}