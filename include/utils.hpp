/*
 * @Author: Jimazeyu
 * @Date: 2022-06-08 09:14:23
 * @LastEditors: Jimazeyu
 * @LastEditTime: 2022-06-08 10:08:24
 * @FilePath: /Stereo_Bighomework/include/utils.hpp
 * @Description: 一些其他函数,包括双目图像显示、绘制特征点匹配关系、3D点云构建等等
 * @Version: 1.0
 */
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <unistd.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <iostream>
using namespace std;

/**
 * @description: 绘制特征点匹配结果
 * @return {*}
 * @author: Jimazeyu
 */
void draw_matches(const cv::Mat &img1, const cv::Mat &img2, const vector<pair<pair<float, float>, pair<float, float>>> &matches)
{
    // 将图像拼接成一张图
    cv::Mat img_matches;
    cv::hconcat(img1, img2, img_matches);
    // 彩色绘制匹配结果
    cv::cvtColor(img_matches, img_matches, cv::COLOR_GRAY2BGR);
    for (auto &match : matches)
    {
        cv::Point2f p1(match.first.first, match.first.second);
        cv::Point2f p2(match.second.first + img1.cols, match.second.second);
        cv::line(img_matches, p1, p2, cv::Scalar(0, 0, 255), 2);
    }
    // resize
    cv::resize(img_matches, img_matches, cv::Size(img_matches.cols / 2, img_matches.rows / 2));
    // 显示匹配结果
    cv::imshow("matches", img_matches);
    while (true)
    {
        if (cv::waitKey(10) == 27)
        {
            cv::destroyAllWindows();
            break;
        }
    }
}

/**
 * @description: 显示左右两幅图像
 * @param {Mat} &left
 * @param {Mat} &right
 * @return {*}
 * @author: Jimazeyu
 */
void show_two_eyes(cv::Mat &left, cv::Mat &right)
{
    // 显示左右相机图像
    cv::Mat img_stitch;
    cv::hconcat(left, right, img_stitch);
    cv::resize(img_stitch, img_stitch, cv::Size(img_stitch.cols / 2, img_stitch.rows / 2));
    cv::imshow("Two cameras", img_stitch);
    while (true)
    {
        if (cv::waitKey(10) == 27)
        {
            cv::destroyAllWindows();
            break;
        }
    }
}

/**
 * @description: 绘制3D点云
 * @param {vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>} &pointcloud
 * @return {*}
 * @author: Jimazeyu
 */
void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud)
{

    if (pointcloud.empty())
    {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }
    // 创建窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);                           // 3d必开，只现实镜头视角的像素
    glEnable(GL_BLEND);                                //颜色混合，透过蓝玻璃看黄玻璃变绿
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); //设置上一行的混合方式

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));
    /*
    进行显示设置。SetBounds函数前四个参数依次表示视图在视窗中的范围（下、上、左、右），最后一个参数是显示的长宽比。
    （0.0, 1.0, 0.0, 1.0）第一个参数0.0表示显示的拍摄窗口的下边在整个GUI中最下面，第二个参数1.0表示上边在GUI的最上面，以此类推。如果在中间就用0.5表示。
    */
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam)); //创建相机视图句柄

    while (pangolin::ShouldQuit() == false)
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //清空颜色和深度缓存。这样每次都会刷新显示，不至于前后帧的颜信息相互干扰。
        d_cam.Activate(s_cam);                              //激活显示并设置状态矩阵
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);               //前三个分别代表红、绿、蓝所占的分量，范围从0.0f~1.0f,最后一个参数是透明度Alpha值,范围也是0.0f~1.0f
        glPointSize(2);                                     //设置点的大小
        glBegin(GL_POINTS);                                 //开始绘制点云
        for (auto &p : pointcloud)
        {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd(); //结束绘制点云
        pangolin::FinishFrame();
        usleep(5000); //一帧5ms
    }
    return;
}

#endif
