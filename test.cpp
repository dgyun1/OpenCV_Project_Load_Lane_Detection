
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

	//관심 영역 범위 계산시 사용
	double poly_bottom_width = 0.85;  //사다리꼴 아래쪽 가장자리 너비 계산을 위한 백분율
	double poly_top_width = 0.07;     //사다리꼴 위쪽 가장자리 너비 계산을 위한 백분율
	double poly_height = 0.4;         //사다리꼴 높이 계산을 위한 백분율
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
	// 흰색/노란색 색상의 범위를 정해 해당되는 차선을 필터링한다.

	Mat output;
	UMat img_hsv;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;
	img_frame.copyTo(output);

	//차선 색깔 범위
	Scalar lower_white = Scalar(200, 200, 200); //흰색 차선 (RGB)
	Scalar upper_white = Scalar(255, 255, 255);
	Scalar lower_yellow = Scalar(10, 100, 100); //노란색 차선 (HSV)
	Scalar upper_yellow = Scalar(40, 255, 255);

	//흰색 필터링
	inRange(output, lower_white, upper_white, white_mask);
	bitwise_and(output, output, white_image, white_mask);

	cvtColor(output, img_hsv, COLOR_BGR2HSV);

	//노란색 필터링
	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
	bitwise_and(output, output, yellow_image, yellow_mask);

	//두 영상을 합친다.
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);
	return output;
}
Mat RoadLineDetector::limit_region(Mat img_edge) {

	// 관심 영역의 가장자리만 감지되도록 마스킹한다.
	// 관심 영역의 가장자리만 표시되는 이진 영상을 반환한다.

	int width = img_edge.cols;
	int height = img_edge.rows;

	Mat output;
	Mat mask = Mat::zeros(height, width, CV_8UC1);

	//관심 영역 정점 계산
	Point points[4]{
		Point((width * (1 - poly_bottom_width)) / 2, height),
		Point((width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_top_width)) / 2, height - height * poly_height),
		Point(width - (width * (1 - poly_bottom_width)) / 2, height)
	};

	//정점으로 정의된 다각형 내부의 색상을 채워 그린다.
	fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));

	//결과를 얻기 위해 edges 이미지와 mask를 곱한다.
	bitwise_and(img_edge, mask, output);
	return output;
}

vector<Vec4i> RoadLineDetector::houghLines(Mat img_mask) {

	// 관심영역으로 마스킹 된 이미지에서 모든 선을 추출하여 반환

	vector<Vec4i> line;

	//확률적용 허프변환 직선 검출 함수
	HoughLinesP(img_mask, line, 1, CV_PI / 180, 20, 10, 20);
	return line;
}

vector<vector<Vec4i>> RoadLineDetector::separateLine(Mat img_edge, vector<Vec4i> lines) {

	// 검출된 모든 허프변환 직선들을 기울기 별로 정렬한다.
	// 선을 기울기와 대략적인 위치에 따라 좌우로 분류한다.
	vector<vector<Vec4i>> output(2);
	Point p1, p2;
	vector<double> slopes;
	vector<Vec4i> final_lines, left_lines, right_lines;
	double slope_thresh = 0.3;

	//검출된 직선들의 기울기를 계산
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		p1 = Point(line[0], line[1]);
		p2 = Point(line[2], line[3]);

		double slope;
		if (p2.x - p1.x == 0)  //코너 일 경우
			slope = 999.0;
		else
			slope = (p2.y - p1.y) / (double)(p2.x - p1.x);

		//기울기가 너무 수평인 선은 제외
		if (abs(slope) > slope_thresh) {
			slopes.push_back(slope);
			final_lines.push_back(line);
		}
	}

	//선들을 좌우 선으로 분류
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

	// 선형 회귀를 통해 좌우 차선 각각의 가장 적합한 선을 찾는다.

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
			//주어진 contour에 최적화된 직선 추출
			fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

			right_m = right_line[1] / right_line[0];  //기울기
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
			//주어진 contour에 최적화된 직선 추출
			fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

			left_m = left_line[1] / left_line[0];  //기울기
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	//좌우 선 각각의 두 점을 계산한다.
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

	// 두 차선이 교차하는 지점(사라지는 점)이 중심점으로부터
	// 왼쪽에 있는지 오른쪽에 있는지로 진행방향을 예측한다.


	string output;

	double x, threshold = 10;

	//두 차선이 교차하는 지점 계산
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

	// 좌우 차선을 경계로 하는 내부 다각형을 투명하게 색을 채운다.
	// 예측 진행 방향 텍스트를 영상에 출력한다.
	// 좌우 차선을 영상에 선으로 그린다.


	vector<Point> poly_points;
	Mat output;
	String outpout;
	img_input.copyTo(output);

	poly_points.push_back(lane[2]);
	poly_points.push_back(lane[0]);
	poly_points.push_back(lane[1]);
	poly_points.push_back(lane[3]);


	fillConvexPoly(output, poly_points, Scalar(20, 245, 201), LINE_AA, 0);  //다각형 색 채우기
	addWeighted(output, 0.3, img_input, 0.7, 0, img_input);  //영상 합하기

	//예측 진행 방향 텍스트를 영상에 출력
	putText(img_input, dir, Point(450, 150), FONT_HERSHEY_PLAIN, 3, Scalar(29, 230, 181), 3, LINE_AA);

	//좌우 차선 선 그리기
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

		VideoCapture video("road.mp4");  //영상 불러오기

		if (!video.isOpened())
		{
			cout << "동영상 파일을 열 수 없습니다. \n" << endl;
			getchar();
			return -1;
		}

		video.read(img_frame);

		if (img_frame.empty()) return -1;
		

		VideoWriter writer;
		int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  //원하는 코덱 선택
		double fps = 1.0;  //프레임
		string filename = "./result.avi";  //결과 파일

		writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
		if (!writer.isOpened()) {
			cout << "출력을 위한 비디오 파일을 열 수 없습니다. \n";
			return -1;
		}

		video.read(img_frame);
		int cnt = 0;

		while (1) {
			// 원본 영상을 읽어온다.
			if (!video.read(img_frame)) break;

			// 흰색, 노란색 범위 내에 있는 것만 필터링하여 차선 후보로 저장한다.
			img_filter = roadLaneDetector.filter_color(img_frame);

			// 영상을 GrayScale 으로 변환한다.
			cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

			// Canny Edge Detection으로 에지를 추출. (잡음 제거를 위한 가우시안 필터링도 포함)
			Canny(img_filter, img_edge, 50, 150);

			// 자동차의 진행방향 바닥에 존재하는 차선만을 검출하기 위한 관심 영역을 지정
			img_mask = roadLaneDetector.limit_region(img_edge);

			// Hough 변환으로 에지에서의 직선 성분을 추출
			lines = roadLaneDetector.houghLines(img_mask);

			if (lines.size() > 0) {
				// 추출한 직선성분으로 좌우 차선에 있을 가능성이 있는 직선들만 따로 뽑아서 좌우 각각 직선을 계산한다.
				// 선형 회귀를 하여 가장 적합한 선을 찾는다.

				separated_lines = roadLaneDetector.separateLine(img_edge, lines);
				separated_lines = roadLaneDetector.separateLine(img_mask, lines);
				lane = roadLaneDetector.regression(separated_lines, img_frame);

				// 진행 방향 예측
				dir = roadLaneDetector.predictDir();

				// 원본 영상 출력
				imshow("video", img_frame);

				// 영상에 최종 차선을 선으로 그리고 내부 다각형을 색으로 채운다. 예측 진행 방향 텍스트를 영상에 출력
				img_result = roadLaneDetector.drawLine(img_frame, lane, dir);
			}
			// 템플릿 이미지를 불러온다.
			Mat img_template1 = imread("car1.PNG", IMREAD_COLOR);
			
			int w = img_template1.cols;
			int h = img_template1.rows;


			// 템플릿 매칭을 한다.
			Mat result_t;
			matchTemplate(img_frame, img_template1, result_t, TM_CCOEFF_NORMED);

			double min_val, max_val;
			Point min_loc, max_loc;
			minMaxLoc(result_t, &min_val, &max_val, &min_loc, &max_loc);
			
			
			// 검출된 템플릿 이미지와 유사한 영역에 사각형을 그려준다.
			Point top_left = max_loc;
			Point bottom_right = Point(top_left.x + w, top_left.y + h);
			rectangle(img_result, top_left, bottom_right, (0, 0, 255), 2);

			// 결과를 동영상 파일로 저장. 캡쳐하여 사진 저장
			writer << img_result;
			if (cnt++ == 15)
				imwrite("img_result.jpg", img_result);  //캡쳐하여 사진 저장


			// 엣지 추출 영상 출력
			imshow("edge", img_edge);

			// 흑백 영상 출력
			imshow("filter", img_filter);

			// 결과 영상 출력
			imshow("result", img_result);

			
		

			//esc 키 종료
			if (waitKey(1) == 27)
				break;

		}

	return 0;
}