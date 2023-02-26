#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>

#include <math.h>

using namespace std;
using namespace cv;

int main()
{
	// 加载级联分类xml文件：
	string car_cascade_name = "cars.xml";    //训练的xml文件
	CascadeClassifier car_cascade;
	car_cascade.load(car_cascade_name);     //加载xml文件

	// 加载视频：
	Mat frame;
	VideoCapture cap;
	string video_name = "dataset_video1.avi";//需要打开的视频名称
	if (!cap.open(video_name))  //判断视频地址是否有视频
	{
		cout << "Cannot open the video file " << video_name << endl;    //打印无视频信息
	}

	// 对视频进行处理
	Mat frame_resized; // 调整视频大小
	Mat gray; // 调整视频为灰色，方便处理
	while (cap.read(frame)) //判断视频是否播放完
	{
		int k = waitKey(30);
		if (k == 27)    //点击ESC退出播放
		{
			int c = waitKey(0);
			break;
		}
		else if (k == 13)  //点击Enter键,暂停
		{
			while (true)
			{
				int c = waitKey(0);
				if (c == 13)    //点击Enter键，继续
				{
					break;
				}
			}
		}

		resize(frame, frame_resized, Size(0, 0), 2, 2);//调整视频大小，放大2倍
		// imshow("Original size", frame);  //输出原视视频
		imshow("Resized", frame_resized);   //输出放大的视频
		cvtColor(frame_resized, gray, COLOR_BGR2GRAY);  //将每帧变为灰度图片
		// 使用 classifier 查找车辆：
		vector<Rect> cars;
		car_cascade.detectMultiScale(gray, cars, 1.14, 2); //检测汽车数量
		if (cars.empty() == true)
		{
			cout << "No cars found" << endl;    //打印结果
			imshow("Cars", frame_resized);      //输出每帧
			continue;     //未检测到汽车，继续循环
		}
		else
		{
			cout << cars.size() << " car(s) found" << endl;   //打印结果
			// imshow("Cars", frame_resized);
		}

		//绘制识别到小车的红色矩形
		for (auto elem : cars)
		{
			Rect car_roi;   //矩形大小位置
			car_roi.x = elem.x;
			car_roi.y = elem.y;
			car_roi.width = elem.width;
			car_roi.height = elem.height;

			rectangle(frame_resized, Point(car_roi.x, car_roi.y),
				Point(car_roi.x + car_roi.width,
					car_roi.y + car_roi.height),
				Scalar(127, 0, 255), 2);//绘制矩形
			imshow("Cars", frame_resized); //显示最终结果
		}
	}

	return 0;
}
