
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

class RoadLineDetector
{
private:
	double img_size, img_center;
	double left_m, right_m;
	Point left_b, right_b;
	bool left_detect = false, right_detect = false;

	//���� ���� ���� ���� ���
	double poly_bottom_width = 0.85;  //��ٸ��� �Ʒ��� �����ڸ� �ʺ� ����� ���� �����
	double poly_top_width = 0.07;     //��ٸ��� ���� �����ڸ� �ʺ� ����� ���� �����
	double poly_height = 0.4;         //��ٸ��� ���� ����� ���� �����
public:
	Mat filter_color(Mat img_frame);
	Mat limit_region(Mat img_edge);
	vector<Vec4i> houghLines(Mat img_mask);
	vector<vector<Vec4i> > separateLine(Mat img_edge, vector<Vec4i> lines);
	vector<Point> regression(vector<vector<Vec4i> > separated_lines, Mat img_input);
	string predictDir();
	Mat drawLine(Mat img_input, vector<Point> lane, string dir);
};
Mat RoadLineDetector::filter_color(Mat img_frame) {
	// ���/����� ������ ������ ���� �ش�Ǵ� ������ ���͸��Ѵ�.

	Mat output;
	UMat img_hsv;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;
	img_frame.copyTo(output);

	//���� ���� ����
	Scalar lower_white = Scalar(200, 200, 200); //��� ���� (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //����� ���� (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	//��� ���͸�
	inRange(output, lower_white, upper_white, white_mask);
	bitwise_and(output, output, white_image, white_mask);

	cvtColor(output, img_hsv, COLOR_BGR2HSV);

	//����� ���͸�
	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(output, output, yellow_image, yellow_mask);

	//�� ������ ��ģ��.
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);
	return output;
}
Mat RoadLineDetector::limit_region(Mat img_edge) {

	// ���� ������ �����ڸ��� �����ǵ��� ����ŷ�Ѵ�.
	// ���� ������ �����ڸ��� ǥ�õǴ� ���� ������ ��ȯ�Ѵ�.

	int width = img_edge.cols;
	int height = img_edge.rows;

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);

	//���� ���� ���� ���
	Point points[4]{
		Point((width * (1 - poly_bottom_width)) / 2, height),
		Point((width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_bottom_width)) / 2, height)
	};

	//�������� ���ǵ� �ٰ��� ������ ������ ä�� �׸���.
	fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));

	//����� ��� ���� edges �̹����� mask�� ���Ѵ�.
	bitwise_and(img_edge, mask, output);
	return output;
}

vector<Vec4i> RoadLineDetector::houghLines(Mat img_mask) {

	// ���ɿ������� ����ŷ �� �̹������� ��� ���� �����Ͽ� ��ȯ

	vector<Vec4i> line;

	//Ȯ������ ������ȯ ���� ���� �Լ�
	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> RoadLineDetector::separateLine(Mat img_edge, vector<Vec4i> lines) {

	// ����� ��� ������ȯ �������� ���� ���� �����Ѵ�.
	// ���� ����� �뷫���� ��ġ�� ���� �¿�� �з��Ѵ�.
	vector<vector<Vec4i>> output(2);
	Point p1, p2;
	vector<double> slopes;
	vector<Vec4i> final_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	//����� �������� ���⸦ ���
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		p1 = Point(line[0], line[1]);
		p2 = Point(line[2], line[3]);

		double slope;
		if (p2.x - p1.x == 0)  //�ڳ� �� ���
			slope = 999.0;
		else
			slope = (p2.y - p1.y) / (double)(p2.x - p1.x);

		//���Ⱑ �ʹ� ������ ���� ����
		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			final_lines.push_back(line);
		}
	}

	//������ �¿� ������ �з�
	img_center = (double)((img_edge.cols / 2));

	for (int i = 0; i < final_lines.size(); i++) {
		p1 = Point(final_lines[i][0], final_lines[i][1]);
		p2 = Point(final_lines[i][2], final_lines[i][3]);

		if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
			right_detect = true;
			right_lines.push_back(final_lines[i]);
		}
		else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center) {
			left_detect = true;
			left_lines.push_back(final_lines[i]);
		}
	}

	output[0] = right_lines;
	output[1] = left_lines;
	return output;
}

vector<Point> RoadLineDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {

	// ���� ȸ�͸� ���� �¿� ���� ������ ���� ������ ���� ã�´�.

	vector<Point> output(4);
	Point p1, p2, p3, p4;
	Vec4d left_line, right_line;
	vector<Point> left_points, right_points;

	if (right_detect) {
		for (auto i : separatedLines[0]) {
			p1 = Point(i[0], i[1]);
			p2 = Point(i[2], i[3]);

			right_points.push_back(p1);
			right_points.push_back(p2);
		}

		if (right_points.size() > 0) {
			//�־��� contour�� ����ȭ�� ���� ����
			fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0];  //����
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	if (left_detect) {
		for (auto j : separatedLines[1]) {
			p3 = Point(j[0], j[1]);
			p4 = Point(j[2], j[3]);

			left_points.push_back(p3);
			left_points.push_back(p4);
		}

		if (left_points.size() > 0) {
			//�־��� contour�� ����ȭ�� ���� ����
			fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //����
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//�¿� �� ������ �� ���� ����Ѵ�.
	//y = m*x + b  --> x = (y-b) / m
	int y1 = img_input.rows;
	int y2 = 470;

	double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
	double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

	double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
	double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_x1, y1);
	output[1] = Point(right_x2, y2);
	output[2] = Point(left_x1, y1);
	output[3] = Point(left_x2, y2);

	return output;
}

string RoadLineDetector::predictDir() {

	// �� ������ �����ϴ� ����(������� ��)�� �߽������κ���
	// ���ʿ� �ִ��� �����ʿ� �ִ����� ��������� �����Ѵ�.


	string output;

	double x, threshold = 10;

	//�� ������ �����ϴ� ���� ���
	x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	if (x >= (img_center - threshold) && x <= (img_center + threshold))
		output = "Safy: Straight";
	else if (x > img_center + threshold)
		output = "Warning: Right Turn";
	else if (x < img_center - threshold)
		output = "Warning: Left Turn";

	return output;
}
Mat RoadLineDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {

	// �¿� ������ ���� �ϴ� ���� �ٰ����� �����ϰ� ���� ä���.
	// ���� ���� ���� �ؽ�Ʈ�� ���� ����Ѵ�.
	// �¿� ������ ���� ������ �׸���.


	vector<Point> poly_points;
	Mat output;
	String outpout;
	img_input.copyTo(output);

	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);


	fillConvexPoly(output, poly_points, Scalar(20, 245, 201), LINE_AA, 0);  //�ٰ��� �� ä���
	addWeighted(output, 0.3, img_input, 0.7, 0, img_input);  //���� ���ϱ�

	//���� ���� ���� �ؽ�Ʈ�� ���� ���
	putText(img_input, dir, Point(450, 150), FONT_HERSHEY_PLAIN, 3, Scalar(29, 230, 181), 3, LINE_AA);

	//�¿� ���� �� �׸���
	line(img_input, lane[0], lane[1], Scalar(0, 0, 255), 5, LINE_AA);
	line(img_input, lane[2], lane[3], Scalar(0, 0, 255), 5, LINE_AA);

	return img_input;
}


int main()
	{

		RoadLineDetector roadLaneDetector;
		Mat img_frame, img_filter, img_edge, img_mask, img_lines, img_result;
		vector<Vec4i> lines;
		vector<vector<Vec4i> > separated_lines;
		vector<Point> lane;
		string dir;

		VideoCapture video("road.mp4");  //���� �ҷ�����

		if (!video.isOpened())
		{
			cout << "������ ������ �� �� �����ϴ�. \n" << endl;
			getchar();
			return -1;
		}

		video.read(img_frame);

		if (img_frame.empty()) return -1;
		

		VideoWriter writer;
		int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //���ϴ� �ڵ� ����
		double fps = 1.0;  //������
		string filename = "./result.avi";  //��� ����

		writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
		if (!writer.isOpened()) {
			cout << "����� ���� ���� ������ �� �� �����ϴ�. \n";
			return -1;
		}

		video.read(img_frame);
		int cnt = 0;

		while (1) {
			// ���� ������ �о�´�.
			if (!video.read(img_frame)) break;

			// ���, ����� ���� ���� �ִ� �͸� ���͸��Ͽ� ���� �ĺ��� �����Ѵ�.
			img_filter = roadLaneDetector.filter_color(img_frame);

			// ������ GrayScale ���� ��ȯ�Ѵ�.
			cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

			// Canny Edge Detection���� ������ ����. (���� ���Ÿ� ���� ����þ� ���͸��� ����)
			Canny(img_filter, img_edge, 50, 150);

			// �ڵ����� ������� �ٴڿ� �����ϴ� �������� �����ϱ� ���� ���� ������ ����
			img_mask = roadLaneDetector.limit_region(img_edge);

			// Hough ��ȯ���� ���������� ���� ������ ����
			lines = roadLaneDetector.houghLines(img_mask);

			if (lines.size() > 0) {
				// ������ ������������ �¿� ������ ���� ���ɼ��� �ִ� �����鸸 ���� �̾Ƽ� �¿� ���� ������ ����Ѵ�.
				// ���� ȸ�͸� �Ͽ� ���� ������ ���� ã�´�.

				separated_lines = roadLaneDetector.separateLine(img_edge, lines);
				separated_lines = roadLaneDetector.separateLine(img_mask, lines);
				lane = roadLaneDetector.regression(separated_lines, img_frame);

				// ���� ���� ����
				dir = roadLaneDetector.predictDir();

				// ���� ���� ���
				imshow("video", img_frame);

				// ���� ���� ������ ������ �׸��� ���� �ٰ����� ������ ä���. ���� ���� ���� �ؽ�Ʈ�� ���� ���
				img_result = roadLaneDetector.drawLine(img_frame, lane, dir);
			}
			// ���ø� �̹����� �ҷ��´�.
			Mat img_template1 = imread("car1.PNG", IMREAD_COLOR);
			
			int w = img_template1.cols;
			int h = img_template1.rows;


			// ���ø� ��Ī�� �Ѵ�.
			Mat result_t;
			matchTemplate(img_frame, img_template1, result_t, TM_CCOEFF_NORMED);

			double min_val, max_val;
			Point min_loc, max_loc;
			minMaxLoc(result_t, &min_val, &max_val, &min_loc, &max_loc);
			
			
			// ����� ���ø� �̹����� ������ ������ �簢���� �׷��ش�.
			Point top_left = max_loc;
			Point bottom_right = Point(top_left.x + w, top_left.y + h);
			rectangle(img_result, top_left, bottom_right, (0, 0, 255), 2);

			// ����� ������ ���Ϸ� ����. ĸ���Ͽ� ���� ����
			writer << img_result;
			if (cnt++ == 15)
				imwrite("img_result.jpg", img_result);  //ĸ���Ͽ� ���� ����


			// ���� ���� ���� ���
			imshow("edge", img_edge);

			// ��� ���� ���
			imshow("filter", img_filter);

			// ��� ���� ���
			imshow("result", img_result);

			
		

			//esc Ű ����
			if (waitKey(1) == 27)
				break;

		}

	return 0;
}