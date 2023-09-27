/*
 * @Author: Jimazeyu
 * @Date: 2022-06-08 08:38:53
 * @LastEditors: Jimazeyu
 * @LastEditTime: 2022-06-08 10:08:25
 * @FilePath: /Stereo_Bighomework/include/featureMatch.hpp
 * @Description: 特征点匹配
 * @Version: 1.0
 */
#ifndef _FEATURE_MATCH_HPP_
#define _FEATURE_MATCH_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include "utils.hpp"
#include "featureDetector.hpp"
#include "featureDescriptor.hpp"
using namespace cv;
using namespace std;

/**
 * @description: 暴力匹配描述子
 * @param {vector<vector<uint32_t>>} &desc1
 * @param {vector<vector<uint32_t>>} &desc2
 * @param {vector<cv::DMatch>} &matches
 * @return {*}
 * @author: Jimazeyu
 */
void BfMatch(const vector<vector<uint32_t>> &desc1, const vector<vector<uint32_t>> &desc2, vector<cv::DMatch> &matches)
{
  const int d_max = 40;

  for (size_t i1 = 0; i1 < desc1.size(); ++i1)
  {
    if (desc1[i1].empty())
      continue;
    cv::DMatch m{(int)i1, 0, 256};
    for (size_t i2 = 0; i2 < desc2.size(); ++i2)
    {
      if (desc2[i2].empty())
        continue;
      int distance = 0;
      for (int k = 0; k < 8; k++)
      {
        // 计算汉明距离
        distance += __builtin_popcount(desc1[i1][k] ^ desc2[i2][k]);
      }
      if (distance < d_max && distance < m.distance)
      {
        m.distance = distance;
        m.trainIdx = i2;
      }
    }
    if (m.distance < d_max)
    {
      matches.push_back(m);
    }
  }
}

/**
 * @description: 使用特征提取算法以及匹配算法获得匹配关系(CV版本,默认ORB实现)
 * @param {Mat} &img1
 * @param {Mat} &img2
 * @return {vector<pair<pair<float, float>} matches
 * @author: Jimazeyu
 */
vector<pair<pair<float, float>, pair<float, float>>> cv_get_matches(cv::Mat &img1, cv::Mat &img2)
{
  // 计算左右图像特征点
  vector<cv::KeyPoint> left_keypoints, right_keypoints;
  // ORB特征检测
  cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
  detector->detect(img1, left_keypoints);
  detector->detect(img2, right_keypoints);
  // 计算左右图像特征点描述子
  cv::Mat left_descriptors, right_descriptors;
  detector->compute(img1, left_keypoints, left_descriptors);
  detector->compute(img2, right_keypoints, right_descriptors);
  // 对左右图像特征点描述子进行匹配
  vector<cv::DMatch> matches;
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  matcher->match(left_descriptors, right_descriptors, matches);
  // 可视化匹配结果
  cv::Mat img_matches;
  cv::drawMatches(img1, left_keypoints, img2, right_keypoints, matches, img_matches);
  cv::resize(img_matches, img_matches, cv::Size(img_matches.cols / 2, img_matches.rows / 2));
  cv::imshow("BruteForce Matches", img_matches);
  while (true)
  {
    if (cv::waitKey(10) == 27)
    {
      cv::destroyAllWindows();
      break;
    }
  }
  // 提取所有匹配点
  vector<cv::Point2f> left_points, right_points;
  for (int i = 0; i < (int)matches.size(); i++)
  {
    // 左图像上的点
    left_points.push_back(left_keypoints[matches[i].queryIdx].pt);
    // 右图像上的点
    right_points.push_back(right_keypoints[matches[i].trainIdx].pt);
  }

  // 通过极线判断匹配是否正确
  vector<pair<pair<float, float>, pair<float, float>>> rectify_matches;
  for (int i = 0; i < left_points.size(); i++)
  {
    auto point_left = left_points[i];
    auto point_right = right_points[i];
    // 如果两个点y坐标小于一定像素，则认为是同一条极线
    if (abs(point_left.y - point_right.y) < 5)
    {
      rectify_matches.push_back(make_pair(make_pair(point_left.x, point_left.y), make_pair(point_right.x, point_right.y)));
    }
  }
  // 画出rectify_matches
  draw_matches(img1, img2, rectify_matches);
  return rectify_matches;
}

/**
 * @description: 使用特征提取算法以及匹配算法获得匹配关系(手写版本)
 * @param {Mat} &img1
 * @param {Mat} &img2
 * @param {string} &method
 * @return {vector<pair<pair<float, float>} matches
 * @author: Jimazeyu
 */
vector<pair<pair<float, float>, pair<float, float>>> get_matches(cv::Mat &img1, cv::Mat &img2, const string &method)
{
  // 计算左右图像特征点
  vector<cv::KeyPoint> keypoints1, keypoints2;
  // 角点检测
  if (method == "ORB")
  {
    FAST(img1, keypoints1, 40);
    FAST(img2, keypoints2, 40);
  }
  // ORB计算描述子
  vector<vector<uint32_t>> descriptor1, descriptor2;
  ComputeORB(img1, keypoints1, descriptor1);
  ComputeORB(img2, keypoints2, descriptor2);
  // 对左右图像特征点描述子进行匹配
  vector<cv::DMatch> matches;
  BfMatch(descriptor1, descriptor2, matches);
  // 可视化匹配结果
  cv::Mat img_matches;
  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
  cv::resize(img_matches, img_matches, cv::Size(img_matches.cols / 2, img_matches.rows / 2));
  cv::imshow("BruteForce Matches", img_matches);
  while (true)
  {
    if (cv::waitKey(10) == 27)
    {
      cv::destroyAllWindows();
      break;
    }
  }
  // 提取所有匹配点
  vector<cv::Point2f> left_points, right_points;
  for (int i = 0; i < (int)matches.size(); i++)
  {
    // 左图像上的点
    left_points.push_back(keypoints1[matches[i].queryIdx].pt);
    // 右图像上的点
    right_points.push_back(keypoints2[matches[i].trainIdx].pt);
  }

  // 通过极线判断匹配是否正确
  vector<pair<pair<float, float>, pair<float, float>>> rectify_matches;
  for (int i = 0; i < left_points.size(); i++)
  {
    auto point_left = left_points[i];
    auto point_right = right_points[i];
    // 如果两个点y坐标小于一定像素，则认为是同一条极线
    if (abs(point_left.y - point_right.y) < 5)
    {
      rectify_matches.push_back(make_pair(make_pair(point_left.x, point_left.y), make_pair(point_right.x, point_right.y)));
    }
  }
  // 画出rectify_matches
  draw_matches(img1, img2, rectify_matches);
  return rectify_matches;
}

#endif