/*
 * @Author: Chenhaopeng
 * @Date: 2022-06-18 21:49:53
 * @LastEditors: Chenhaopeng
 * @LastEditTime: 2022-06-18 21:49:53
 * @FilePath: /Stereo_Bighomework/include/camera_calibration.hpp
 * @Description: 单目相机标定、双目相机标定及图像矫正
 * @Version: 1.0
 */
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv; 
using namespace std;

//画标定板的函数
void drawChessBoard(int blocks_per_row = 11, int blocks_per_col = 8, int block_size = 75);// 11  8  75  
//图片像素
const int imageWidth = 640;
const int imageHeight = 480;
//横向的角点数目
const int points_per_row = 10;
//纵向的角点数目
const int points_per_col = 7;
//总的角点数目
const int boardCorner = points_per_row * points_per_col;//标定板每行每列角点个数，共10*7个角点
//相机标定时需要采用的图像帧数
const int frameNumber = 8;
//标定板黑白格子的大小 单位是mm
const int squareSize = 75;
//标定板的总内角点
const Size corner_size = Size(points_per_row, points_per_col);
Size image_size = Size(imageWidth, imageHeight);//图片尺寸

Mat R, T, E, F;
//R旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<Mat> rvecs; //R
vector<Mat> tvecs; //T
//左边摄像机所有照片角点的坐标集合（虽然左右是同一个相机）
vector<vector<Point2f>> imagePointL;
//右边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR;
//各图像的角点的实际的物理坐标集合
vector<vector<Point3f>> objRealPoint;
//左边摄像机某一照片角点坐标集合
vector<Point2f> cornerL;
//右边摄像机某一照片角点坐标集合
vector<Point2f> cornerR;
//为左右图片创建RGB和灰度图的矩阵
Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
//校正旋转矩阵R，投影矩阵P，重投影矩阵Q
Mat Rl, Rr, Pl, Pr, Q;
//映射表
Mat mapLx, mapLy, mapRx, mapRy;
Rect validROIL, validROIR;//留个矩形画图备用

/**
 * @description: 计算标定板上模块的实际物理坐标
 * @param {vector<vector<Point3f>>} & obj
 * @param {int} points_per_row
 * @param {int} points_per_col
 * @param {int} imgNumber
 * @param {int} squareSize
 * @return {}
 * @author: Chenhaopeng
 */
void RealPoint(vector<vector<Point3f>>& obj, int points_per_row, int points_per_col, int imgNumber, int squareSize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < points_per_col; rowIndex++)
    {
        for (int colIndex = 0; colIndex < points_per_row; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}


int main(int argc, char** argv)
{
    //drawChessBoard();//画标定板
    //先进行单目相机的标定
    ifstream fin("file_images.txt"); //读取标定图片的路径，与cpp程序在同一路径下
    if (!fin)                              //检测是否读取到文件
    {
        cerr << "没有找到文件" << endl;
        return -1;
    }
    ofstream fout("calibration_result.txt"); //输出结果保存在此文本文件下
    //依次读取每一幅图片，从中提取角点
    cout << "开始提取角点……" << endl; 
    int image_nums = 0;  //图片数量
    vector<Point2f> points_per_image;               //缓存每幅图检测到的角点
    vector<vector<Point2f>> points_all_images;      //用一个二维数组保存检测到的所有角点
    string image_file_name;                         //声明一个文件名的字符串
    while (getline(fin, image_file_name))           //逐行读取，将行读入字符串
    {
        image_nums++;
        Mat image_raw = imread(image_file_name);        //读入图片
        if (image_nums == 1)//确认下标定图片的尺寸
        {
            image_size.width = image_raw.cols;  //图像的宽对应着列数
            image_size.height = image_raw.rows; //图像的高对应着行数
            cout << "image_size.width = " << image_size.width << endl;
            cout << "image_size.height = " << image_size.height << endl;
        }
        //角点检测部分
        Mat image_gray;                               //存储灰度图的矩阵
        cvtColor(image_raw, image_gray, CV_RGB2GRAY); //将RBG图转化为灰度图
        // 提取角点
        bool success = findChessboardCorners(image_gray, corner_size, points_per_image);
        if (!success)
        {
            cout << "can not find the corners " << endl;
            exit(1);
        }
        else
        {
            //亚像素精确化（两种方法）
            find4QuadCornerSubpix(image_gray, points_per_image, Size(5, 5));  //亚像素角点            
            points_all_images.push_back(points_per_image); //保存亚像素角点
            //在图中画出角点位置
            //角点可视化
            drawChessboardCorners(image_raw, corner_size, points_per_image, success); //将角点连线
            //调试用查看角点连线效果
            //imshow("Camera calibration", image_raw);
            //waitKey(0); //等待按键输入
        }
    }
    destroyAllWindows();//清空界面
    //输出图像数目
    int image_sum_nums = points_all_images.size();
    cout << "image_sum_nums = " << image_sum_nums << endl;
    //开始相机标定
    Size block_size(21, 21);                            //每个小方格实际大小, 只会影响最后求解的平移向量t
    Mat camera_K(3, 3, CV_32FC1,Scalar::all(0));        //内参矩阵3*3
    Mat distCoeffs(1, 5, CV_32FC1,Scalar::all(0));      //畸变矩阵1*5
    vector<Mat> rotationMat;                            //旋转矩阵
    vector<Mat> translationMat;                         //平移矩阵
    //初始化角点三维坐标,从左到右,从上到下!!!
    vector<Point3f> points3D_per_image;
    for (int i = 0; i < corner_size.height; i++)
    {
        for (int j = 0; j < corner_size.width; j++)
        {
            points3D_per_image.push_back(Point3f(block_size.width * j, block_size.height * i, 0));
        }
    }
    vector<vector<Point3f>> points3D_all_images(image_nums, points3D_per_image);        //保存所有图像角点的三维坐标, z=0

    int point_counts = corner_size.area(); //每张图片上角点个数
    /* calibrateCamera函数使用说明：
     * points3D_all_images: 真实三维坐标
     * points_all_images: 提取的角点
     * image_size: 图像尺寸
     * camera_K : 内参矩阵K
     * distCoeffs: 畸变参数
     * rotationMat: 每个图片的旋转向量
     * translationMat: 每个图片的平移向量
     */
     //相机标定
    calibrateCamera(points3D_all_images, points_all_images, image_size, camera_K, distCoeffs, rotationMat, translationMat, 0);
    //对标定结果进行评价
    double total_err = 0.0;               //所有图像平均误差总和
    double err = 0.0;                     //每幅图像的平均误差
    vector<Point2f> points_reproject;     //重投影点
    cout << "\n\t每幅图像的标定误差:\n";
    fout << "每幅图像的标定误差：\n";
    for (int i = 0; i < image_nums; i++)
    {
        vector<Point3f> points3D_per_image = points3D_all_images[i];
        //通过之前标定得到的相机内外参，对三维点进行重投影
        projectPoints(points3D_per_image, rotationMat[i], translationMat[i], camera_K, distCoeffs, points_reproject);
        //计算两者之间的误差
        vector<Point2f> detect_points = points_all_images[i];  //提取到的图像角点
        Mat detect_points_Mat = Mat(1, detect_points.size(), CV_32FC2); //变为矩阵
        Mat points_reproject_Mat = Mat(1, points_reproject.size(), CV_32FC2);  //2通道保存投影角点的像素坐标
        for (int j = 0; j < detect_points.size(); j++)//将提取的角点信息一一输入矩阵
        {
            detect_points_Mat.at<Vec2f>(0, j) = Vec2f(detect_points[j].x, detect_points[j].y);
            points_reproject_Mat.at<Vec2f>(0, j) = Vec2f(points_reproject[j].x, points_reproject[j].y);
        }
        err = norm(points_reproject_Mat, detect_points_Mat, NormTypes::NORM_L2);//计算误差
        total_err += err /= point_counts;//总误差
        cout << "第" << i + 1 << "幅图像的平均误差为： " << err << "像素" << endl;
        fout << "第" << i + 1 << "幅图像的平均误差为： " << err << "像素" << endl;
    }
    cout << "总体平均误差为： " << total_err / image_nums << "像素" << endl;
    fout << "总体平均误差为： " << total_err / image_nums << "像素" << endl;

    //将标定结果写入txt文件，便于查看
    Mat rotate_Mat = Mat(3, 3, CV_32FC1, Scalar::all(0)); //保存旋转矩阵
    cout << "\n相机内参数矩阵:" << endl;
    cout << camera_K << endl << endl;
    fout << "\n相机内参数矩阵:" << endl;
    fout << camera_K << endl << endl;
    cout << "畸变系数：\n";
    cout << distCoeffs << endl << endl << endl;
    fout << "畸变系数：\n";
    fout << distCoeffs << endl << endl << endl;
    for (int i = 0; i < image_nums; i++)
    {
        Rodrigues(rotationMat[i], rotate_Mat); //将旋转向量通过罗德里格斯公式转换为旋转矩阵
        fout << "第" << i + 1 << "幅图像的旋转矩阵为：" << endl;
        fout << rotate_Mat << endl;
        fout << "第" << i + 1 << "幅图像的平移向量为：" << endl;
        fout << translationMat[i] << endl
            << endl;
    }
    fout << endl;
    fout.close();
    //图像校正之后，会对图像进行裁剪
    //左相机的内参矩阵
    Mat cameraMatrixL = camera_K;
    //获得的畸变参数
    Mat distCoeffL = distCoeffs;
    //右相机的内参矩阵
    Mat cameraMatrixR = camera_K;//没有双目摄像头，只能拿一个摄像头凑数
    Mat distCoeffR = distCoeffs;
    Mat img;
    int count = 0;
    while (count < frameNumber)
    {
        char filename[100];
        /*读取左边的图像*/
        rgbImageL = imread("photo//9.png", IMREAD_COLOR);//这次输入的是用来双目标定的图片
        imshow("chessboardL", rgbImageL);
        cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);//转换成灰度图
        /*读取右边的图像*/
        rgbImageR = imread("photo//8.png", IMREAD_COLOR);
        cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);
        //再次角点检测
        bool FindL, FindR;
        FindL = findChessboardCorners(rgbImageL, corner_size, cornerL);
        FindR = findChessboardCorners(rgbImageR, corner_size, cornerR);
        if (FindL == true && FindR == true)
        {
            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));//亚像素角点检测
            drawChessboardCorners(rgbImageL, corner_size, cornerL, FindL);
            imshow("chessboardL", rgbImageL);//显示左角点连线后的图片
            imagePointL.push_back(cornerL);

            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, corner_size, cornerR, FindR);
            imshow("chessboardR", rgbImageR);//显示右角点连线后的图片
            imagePointR.push_back(cornerR);
            count++;
        }
        else
        {
            cout << "the image is bad please try again" << endl;
        }
        if (waitKey(1) == 'q')
        {
            break;
        }
    }

    //计算实际的校正点的三维坐标，根据实际标定格子的大小来设置
    RealPoint(objRealPoint, points_per_row, points_per_col, frameNumber, squareSize);
    cout << "标定成功" << endl;

    //标定双目摄像头
    /*stereoCalibrate参数说明：
    * objectPoints- vector<point3f> 型的数据结构，存储标定角点在世界坐标系中的位置
    * imagePoints1- vector<vector<point2f>> 型的数据结构，存储标定角点在第一个摄像机下的投影后的亚像素坐标
    * imagePoints2- vector<vector<point2f>> 型的数据结构，存储标定角点在第二个摄像机下的投影后的亚像素坐标
    * cameraMatrix1-输入/输出型的第一个摄像机的相机矩阵。如果CV_CALIB_USE_INTRINSIC_GUESS , CV_CALIB_FIX_ASPECT_RATIO ,CV_CALIB_FIX_INTRINSIC , or CV_CALIB_FIX_FOCAL_LENGTH其中的一个或多个标志被设置，该摄像机矩阵的一些或全部参数需要被初始化
    * distCoeffs1-第一个摄像机的输入/输出型畸变向量。根据矫正模型的不同，输出向量长度由标志决定
    * cameraMatrix2-输入/输出型的第二个摄像机的相机矩阵。参数意义同第一个相机矩阵相似
    * distCoeffs2-第一个摄像机的输入/输出型畸变向量。根据矫正模型的不同，输出向量长度由标志决定
    * imageSize-图像的大小
    * R-输出型，第一和第二个摄像机之间的旋转矩阵
    * T-输出型，第一和第二个摄像机之间的平移矩阵
    * E-输出型，基本矩阵
    * F-输出型，基础矩阵
    * term_crit-迭代优化的终止条件
    * flag-
    *  CV_CALIB_FIX_INTRINSIC 如果该标志被设置，那么就会固定输入的cameraMatrix和distCoeffs不变，只求解R,T,E,F
    *  CV_CALIB_USE_INTRINSIC_GUESS 根据用户提供的cameraMatrix和distCoeffs为初始值开始迭代
    *  CV_CALIB_FIX_PRINCIPAL_POINT 迭代过程中不会改变主点的位置
    *  CV_CALIB_FIX_FOCAL_LENGTH 迭代过程中不会改变焦距
    *  CV_CALIB_SAME_FOCAL_LENGTH 强制保持两个摄像机的焦距相同
    *  CV_CALIB_ZERO_TANGENT_DIST 切向畸变保持为零
    *  CV_CALIB_FIX_K1,...,CV_CALIB_FIX_K6 迭代过程中不改变相应的值。如果设置了 CV_CALIB_USE_INTRINSIC_GUESS 将会使用用户提供的初始值，否则设置为零
    *  CV_CALIB_RATIONAL_MODEL 畸变模型的选择，如果设置了该参数，将会使用更精确的畸变模型，distCoeffs的长度就会变成8
    */
    double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
        cameraMatrixL, distCoeffL,
        cameraMatrixR, distCoeffR,
        image_size, R, T, E, F, CALIB_USE_INTRINSIC_GUESS,
        TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));

    cout << "立体标定误差RMS为 " << rms << endl;//输出标定误差
    //立体矫正函数计算出矫正参数
    /*stereoRectify参数说明：
    * cameraMatrix1-第一个摄像机的摄像机矩阵
    * distCoeffs1-第一个摄像机的畸变向量
    * cameraMatrix2-第二个摄像机的摄像机矩阵
    * distCoeffs1-第二个摄像机的畸变向量
    * imageSize-图像大小
    * R- stereoCalibrate() 求得的R矩阵
    * T- stereoCalibrate() 求得的T矩阵
    * R1-输出矩阵，第一个摄像机的校正变换矩阵（旋转变换）
    * R2-输出矩阵，第二个摄像机的校正变换矩阵（旋转矩阵）
    * P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵
    * P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵
    * Q-4*4的深度差异映射矩阵
    * flags-可选的标志有两种零或者 CV_CALIB_ZERO_DISPARITY ,如果设置 CV_CALIB_ZERO_DISPARITY 的话，该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大
    * alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。如果设置为0，那么校正后图像只有有效的部分会被显示（没有黑色的部分），如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。
    * newImageSize-校正后的图像分辨率，默认为原分辨率大小。
    * validPixROI1-可选的输出参数，Rect型数据。其内部的所有像素都有效
    * validPixROI2-可选的输出参数，Rect型数据。其内部的所有像素都有效
    */
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, image_size, R, T, Rl,
        Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, image_size, &validROIL, &validROIR);

    /*像素去畸变
    * initUndistortRectifyMap函数参数说明：
    * cameraMatrix——输入的摄像头内参数矩阵（3X3矩阵）
    * distCoeffs——输入的摄像头畸变系数矩阵（5X1矩阵）
    * newCameraMatrix——输入的校正后的3X3摄像机矩阵
    * size——摄像头采集的无失真图像尺寸
    * m1type——map1的数据类型，可以是CV_32FC1或CV_16SC2
    * map1——输出的X坐标重映射参数
    * map2——输出的Y坐标重映射参数
    */
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, image_size, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, image_size, CV_32FC1, mapRx, mapRy);
    //输入需要矫正的图片
    Mat photoleft = imread("photo//4.png");
    Mat photoright = imread("photo//3.png");
    imshow("Recitify Before", photoleft);
    Mat rectifyImageL, rectifyImageR;
    /* remap函数说明：一幅图像中某位置的像素放置到另一个图片指定位置。
    * 参数说明：
    * src——输入图像，即原图像，需要单通道8位或者浮点类型的图像
    * dst(c++)——输出图像，即目标图像，需和原图形一样的尺寸和类型
    * map1——它有两种可能表示的对象：（1）表示点（x,y）的第一个映射；（2）表示CV_16SC2，CV_32FC1等
    * map2——有两种可能表示的对象：（1）若map1表示点（x,y）时，这个参数不代表任何值；（2）表示 CV_16UC1，CV_32FC1类型的Y值
    * intermap2polation——插值方式，有四中插值方式：
    * （1）INTER_NEAREST——最近邻插值
    * （2）INTER_LINEAR——双线性插值（默认）
    * （3）INTER_CUBIC——双三样条插值（默认）
    * （4）INTER_LANCZOS4——lanczos插值（默认）
    * intborderMode——边界模式，默认BORDER_CONSTANT
    * borderValue——边界颜色，默认Scalar()黑色
    */
    remap(photoleft, rectifyImageL, mapLx, mapLy, INTER_LINEAR);//重映射
    remap(photoright, rectifyImageR, mapRx, mapRy, INTER_LINEAR);
    //经过remap之后，左右相机的图像已经共面并且行对准了
    imshow("rectifyImageL", rectifyImageL);//显示矫正后图像
    imshow("rectifyImageR", rectifyImageR);
    imwrite("rectifyImageL.png", rectifyImageL);//输出矫正后图像
    imwrite("rectifyImageR.png", rectifyImageR);
    //保存及输出数据
    FileStorage fs("intrisics.txt", FileStorage::WRITE);//内参数据
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "左相机内参矩阵：" << cameraMatrixL << endl << "左相机畸变系数：" << distCoeffL << endl << "右相机内参矩阵：" << cameraMatrixR << endl << "右相机畸变系数：" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!" << endl;
    }
    fs.open("extrinsics.txt", FileStorage::WRITE);//外参数据
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr" << Rr << endl << "Pl" << Pl << endl << "Pr" << Pr << endl << "Q" << Q << endl;
        fs.release();
    }
    else
    {
        cout << "Error: can not save the extrinsic parameters\n";
    }
    //显示校正结果
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(image_size.width, image_size.height);
    w = cvRound(image_size.width * sf);
    h = cvRound(image_size.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(0, 0, w, h));
    resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
    Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf),
        cvRound(validROIL.width * sf), cvRound(validROIL.height * sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);

    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));
    resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

    imshow("rectified", canvas);//显示最终图像
    imwrite("rectified.png", canvas);//输出最终图像
    cout << "wait key" << endl;
    waitKey(0);
    return 0;
}



////在电脑上画标定板
////函数声明,默认每行11个block, 每列8个block, block大小为75个像素. 也就是10*7个内点
//void drawChessBoard(int blocks_per_row, int blocks_per_col, int block_size)
//{
//    //blocks_per_row=11 //每行11个格子,也就是10个点
//    //blocks_per_col=8  //每列8个格子,也就是7个点
//    //block_size=75     //每个格子的像素大小
//    Size board_size = Size(block_size * blocks_per_row, block_size * blocks_per_col);
//    Mat chessboard = Mat(board_size, CV_8UC1);
//    unsigned char color = 0;
//    for (int i = 0; i < blocks_per_row; i++)
//    {
//        color = ~color;
//        for (int j = 0; j < blocks_per_col; j++)
//        {
//            chessboard(Rect(i * block_size, j * block_size, block_size, block_size)).setTo(color);
//            color = ~color;
//        }
//    }
//    Mat chess_board = Mat(board_size.height + 100, board_size.width + 100, CV_8UC1, Scalar::all(256)); //上下左右留出50个像素空白
//    chessboard.copyTo(chess_board.rowRange(50, 50 + board_size.height).colRange(50, 50 + board_size.width));
//    imshow("chess_board", chess_board);
//    imwrite("chess_board.png", chess_board);
//    waitKey(-1);
//    destroyAllWindows();
//}


