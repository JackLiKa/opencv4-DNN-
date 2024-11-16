#include<opencv2/opencv.hpp>
#include<bits/stdc++.h>
using namespace cv;
using namespace std;
int main()
{

	//模型加载
	string pb_file_path = "pb/opencv_face_detector_uint8.pb";
	string pbtxt_file_path = "pb/opencv_face_detector.pbtxt";
	dnn::Net net = dnn::readNetFromTensorflow(pb_file_path, pbtxt_file_path);
	VideoCapture video(0);//Read Video
	Mat frame;

	//作用于Save
	int frame_width=video.get(CAP_PROP_FRAME_WIDTH);//获取视频的宽度
	int frame_height=video.get(CAP_PROP_FRAME_HEIGHT);//获取视频的高度
	int count=video.get(CAP_PROP_FRAME_COUNT);//视频总的帧数
	//fps是衡量处理视频的能力
	double fps  = video.get(CAP_PROP_FPS);

	VideoWriter writer("video/demo.mp4", video.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);

	int start = -1;
		Retest:
	cout << "\n\n\n\n\n";
	std::cout << "视频宽" << frame_width << std::endl;
	std::cout << "视频高" << frame_height << std::endl;
	std::cout << "视频FPS" << fps << std::endl;
	std::cout << "视频帧数" << count << std::endl;
	cout << "\n\n\n\n\n";

	cout << "这里是文档：此识别的视频将会保存在与源文件相同的位置。输入1开始进入功能，在人脸识别时可按Esc退出人脸识别并重新该功能" << endl<<"输入1进入功能：";
	cin >> start;
	if (start != 1)goto Retest;
	while (1)
	{
		video.read(frame);
		if (frame.empty()) { break; }
		Mat blob = dnn::blobFromImage(frame, 1.0, Size(300, 300), Scalar(104, 177, 123), false, false);
		net.setInput(blob);
		Mat probs = net.forward();
		//1x1xNx7
		Mat detectMat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());

		for (int row = 0; row < detectMat.rows; ++row)
		{
			float conf = detectMat.at<float>(row, 2);
			if (conf > 0.5)
			{
				float x1 = detectMat.at<float>(row, 3) * frame.cols;
				float y1 = detectMat.at<float>(row, 4) * frame.rows;
				float x2 = detectMat.at<float>(row, 5) * frame.cols;
				float y2 = detectMat.at<float>(row, 6) * frame.rows;
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 255, 255), 2, 8);
				cv::imshow("DNN 人脸检测演示", frame);
				writer.write(frame);//将帧写入保存
			}
		}

		auto c = waitKey(1);
		if (c == 27)
		{
			break;
		}
	}
	waitKey(0);
	cv::destroyAllWindows();
	video.release();//释放相机资源
	writer.release();//释放缓存
	goto Retest;
	return 0;
}