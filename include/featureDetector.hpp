/*
 * @Author: Jimazeyu
 * @Date: 2022-06-08 08:38:53
 * @LastEditors: Jimazeyu
 * @LastEditTime: 2022-06-08 10:05:14
 * @FilePath: /Stereo_Bighomework/include/featureDetector.hpp
 * @Description: 特征点检测
 * @Version: 1.0
 */
#ifndef _FEATURE_DETECTOR_HPP_
#define _FEATURE_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include "FASTCorner.h"
#include "HarrisCorner.h"

using namespace cv;
using namespace std;


/**
 * @description: FAST特征点提取
 * @param {Mat} &img
 * @param {vector<cv::KeyPoint>} &keypoints
 * @param {int} threshold
 * @return {*}
 * @author: Jimazeyu
 */
void FAST(const cv::Mat &img, vector<cv::KeyPoint> &keypoints, int threshold)
{
  // cv::FAST(img, keypoints, threshold, true);
  Mat dst;
  myFastCornerDetection(img,dst,threshold);
  myConvertFASTMatToCVKeyPointVector(dst, keypoints);
}

#endif