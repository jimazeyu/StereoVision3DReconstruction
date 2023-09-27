/*
 * @Author: Jimazeyu
 * @Date: 2022-06-06 07:17:11
 * @LastEditors: Jimazeyu
 * @LastEditTime: 2022-06-08 09:47:22
 * @FilePath: /Stereo_Bighomework/stereoSparse.cpp
 * @Description: 稀疏建图
 * @Version: 1.0
 */

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>
#include "include/utils.hpp"
#include "include/featureMatch.hpp"
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
  // 显示左右相机图像
  show_two_eyes(left, right);

  // 使用特征匹配算法获得匹配结果
  auto good_matches = get_matches(left, right, "ORB");

  // 点云数组
  vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud;
  // 便利像素点，利用disparity和相机内参计算点深度信息，将点存入数组中
  for (auto match : good_matches)
  {
    auto point_left = match.first;
    auto point_right = match.second;
    // 计算disparity
    double disparity = point_left.first - point_right.first;
    // 计算点深度信息
    double depth = (b * fx) / disparity;
    // 获取灰度信息
    double gray_left = left.at<uchar>((int)point_left.second, (int)point_left.first) / 255.0;
    // 将点存入数组中
    Vector4d point = Vector4d(0, 0, 0, gray_left); // 前三维为xyz,第四维为颜色
    double x = (point_left.first - cx) / fx;
    double y = (point_left.second - cy) / fy;
    point[0] = x * depth;
    point[1] = y * depth;
    point[2] = depth;
    pointcloud.push_back(point);
  }

  // 画出点云
  showPointCloud(pointcloud);
  return 0;
}
