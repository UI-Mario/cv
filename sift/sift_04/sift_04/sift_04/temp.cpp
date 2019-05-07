#include "stdio.h"
#include <opencv2/opencv.hpp>
#include"pre.h"
#include"ach.h"
#include"step.h"

using namespace cv;

sift *sift1 = new sift();
sift *sift2 = new sift();
Res r1, r2;

int main()
{
	r1.img = imread(img_path1);
	r2.img = imread(img_path2);

	resize(r1.img, r1.img, Size(r1.img.rows / 2, r1.img.cols / 2));
	resize(r2.img, r2.img, Size(r2.img.rows / 2, r2.img.cols / 2));

	r1.sift = sift1;
	r2.sift = sift2;

	process(r1);
	process(r2);

	Mat img1 = r1.img.clone();
	Mat img2 = r2.img.clone();
	sift1 = r1.sift;
	sift2 = r2.sift;
	imshow("1", r1.img);
	imshow("2", r2.img);


	int row1 = img1.rows, col1 = img1.cols, row2 = img2.rows, col2 = img2.cols;
	int row = max(row1, row2), col = col1 + col2;
	Mat img = Mat(row, col, CV_8UC3);
	for (int r = 0; r < row1; r++) {
		for (int c = 0; c < col1; c++) {
			img.ptr<uchar>(r)[c * 3 + 0] = img1.ptr<uchar>(r)[c * 3 + 0];
			img.ptr<uchar>(r)[c * 3 + 1] = img1.ptr<uchar>(r)[c * 3 + 1];
			img.ptr<uchar>(r)[c * 3 + 2] = img1.ptr<uchar>(r)[c * 3 + 2];
		}
		for (int c = col1 ; c < col1 + col2 ; c++) {
			img.ptr<uchar>(r)[c * 3 + 0] = img2.ptr<uchar>(r)[(c - col1 ) * 3 + 0];
			img.ptr<uchar>(r)[c * 3 + 1] = img2.ptr<uchar>(r)[(c - col1 ) * 3 + 1];
			img.ptr<uchar>(r)[c * 3 + 2] = img2.ptr<uchar>(r)[(c - col1 ) * 3 + 2];
			}
	}

	namedWindow("temp");
	imshow("temp", img);

	
	
	waitKey(0);
	destroyAllWindows();
	sift1->release();
	sift2->release();

	return 0;
}

