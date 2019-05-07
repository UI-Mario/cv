#pragma once
#include "stdio.h"
#include <opencv2/opencv.hpp>
#include <string.h>
#include <math.h>
#include <cxcore.h>  
#include <highgui.h>  
#include <vector>


#define NUMSIZE 2  
#define GAUSSKERN 3.5
#define PI 3.14159265358979323846  
#define INITSIGMA 0.5  
#define SIGMA sqrt(3)  
#define SCALESPEROCTAVE 2
#define MAXOCTAVES 4  
#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
#define peakRelThresh 0.8  
#define LEN 128  //维数

using namespace cv;
using namespace std;

struct ImageLevel { 
	float levelsigma;
	int levelsigmalength;
	float absolute_sigma;
	Mat Level;
};

struct ImageOctave {
	int row, col;          
	float subsample;
	ImageLevel *Octave;
};

struct Keypoint
{
	float row, col; 
	float sx, sy; 
	int octave, level;
	float scale, ori, mag; 
	float *descrip;
	Keypoint *next;
};

struct matchPoint {
	Point2i r1, r2;
	matchPoint(Point2i pt1, Point2i pt2) {
		r1 = pt1;
		r2 = pt2;
	}
};


void expand_kd_node_subtree(struct kd_node* kd_node);  //递归法建立KD树
void assign_part_key(struct kd_node* kd_node);  //计算节点数据的最大方差对应的维数ki，和中值kv
void partition_features(struct kd_node* kd_node);
/*
kd_root为创建好的KD树，feat为要查询的特征点
k为要找到的近邻节点数，SIFT中选取2
nbrs存储查询到的k个近邻数据
max_nn_chkes为最大提取队列次数，即超时限制
成功返回找到的近邻数据个数，否则返回-1
*/
int kdtree_bbf_knn(struct kd_node* kd_root, struct feature* feat, int k,
	struct feature*** nbrs, int max_nn_chks);

void smooth_ori_hist(double* hist, int n);
float getPixelBI(Mat im, float col, float row);
void normalizeVec(float* vec, int dim);   
Mat GaussianKernel2D(float sigma); 
void normalizeMat(Mat mat);
float* GaussianKernel1D(float sigma, int dim);
float GetVecNorm(float* vec, int dim);
float ConvolveLocWidth(float* kernel, int dim, Mat ori, int x, int y);
void Convolve1DWidth(float* kern, int dim, Mat ori, Mat dst);
float ConvolveLocHeight(float* kernel, int dim, Mat ori, int x, int y);
void Convolve1DHeight(float* kern, int dim, Mat ori, Mat dst);
int BlurImage(Mat ori, Mat dst, float sigma);
