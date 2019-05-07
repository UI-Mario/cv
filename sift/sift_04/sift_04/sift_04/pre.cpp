#include "stdio.h"
#include <opencv2/opencv.hpp>
#include"pre.h"

//平滑减少突变影响
void smooth_ori_hist(double* hist, int n)
{
	double prev, tmp, h0 = hist[0];
	int i;

	prev = hist[n - 1];
	for (i = 0; i < n; i++)
	{
		tmp = hist[i];
		hist[i] = 0.25 * prev + 0.5 * hist[i] +
			0.25 * ((i + 1 == n) ? h0 : hist[i + 1]);
		prev = tmp;
	}
}
 
void doubleSizeImageColor(Mat im, Mat imnew)
	{
		int w = im.cols * 2;
		int h = im.rows * 2;
		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				imnew.ptr<float>(r)[3 * c] = (float)im.ptr<float>(r / 2)[3 * (c / 2)];
				imnew.ptr<float>(r)[3 * c + 1] = (float)im.ptr<float>(r / 2)[3 * (c / 2) + 1];
				imnew.ptr<float>(r)[3 * c + 2] = (float)im.ptr<float>(r / 2)[3 * (c / 2) + 2];
			}
		}
	}
 
float getPixelBI(Mat im, float col, float row)
	{
		int irow = (int)row, icol = (int)col;   //实部
		float rfrac, cfrac;                     //虚部
		int width = im.cols;
		int height = im.rows;
		if (irow < 0 || irow >= height
			|| icol < 0 || icol >= width)
			return 0;
		if (row > height - 1)
			row = height - 1;
		if (col > width - 1)
			col = width - 1;
		rfrac = (row - (float)irow);
		cfrac = (col - (float)icol);

		float row1 = 0, row2 = 0;
		if (cfrac > 0) {
			row1 = (1 - cfrac)*im.ptr<float>(irow)[icol] + cfrac * im.ptr<float>(irow)[icol + 1];
		}
		else {
			row1 = im.ptr<float>(irow)[icol];
		}
		if (rfrac > 0) {
			if (cfrac > 0) {
				row2 = (1 - cfrac)*im.ptr<float>(irow + 1)[icol] + cfrac * im.ptr<float>(irow + 1)[icol + 1];
			}
			else row2 = im.ptr<float>(irow + 1)[icol];
		}
		else {
			return row1;
		}
		return ((1 - rfrac)*row1 + rfrac * row2);
	}

void normalizeMat(Mat mat)
	{
		float sum = 0;

		for (unsigned int r = 0; r < mat.rows; r++)
			for (unsigned int c = 0; c < mat.cols; c++)
				sum += mat.ptr<float>(r)[c];
		for (unsigned int r = 0; r < mat.rows; r++)
			for (unsigned int c = 0; c < mat.cols; c++)
				mat.ptr<float>(r)[c] = mat.ptr<float>(r)[c] / sum;
	}

void normalizeVec(float* vec, int dim)
	{
		unsigned int i;
		float sum = 0;
		for (i = 0; i < dim; i++)
			sum += vec[i];
		for (i = 0; i < dim; i++)
			vec[i] /= sum;
	}


float GetVecNorm(float* vec, int dim)
	{
		float sum = 0.0;
		for (unsigned int i = 0; i<dim; i++)
			sum += vec[i] * vec[i];
		return sqrt(sum);
	}

float* GaussianKernel1D(float sigma, int dim)
	{
		float *kern = (float*)malloc(dim * sizeof(float));
		float s2 = sigma * sigma;
		int c = dim / 2;
		float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);
		double v;
		for (int i = 0; i < (dim + 1) / 2; i++)
		{
			v = m * exp(-(1.0*i*i) / (2.0 * s2));
			kern[c + i] = v;
			kern[c - i] = v;
		}
		return kern;
	}


Mat GaussianKernel2D(float sigma)
	{
		int dim = (int)max(3.0, 2.0 * GAUSSKERN *sigma + 1.0);
		if (dim % 2 == 0)
			dim++;
		Mat mat(dim, dim, CV_32FC1);
		float s2 = sigma * sigma;
		int c = dim / 2;
		float m = 1.0 / (sqrt(2.0 * CV_PI) * sigma);    //前方系数
		for (int i = 0; i < (dim + 1) / 2; i++)
		{
			for (int j = 0; j < (dim + 1) / 2; j++)
			{
				float v = m * exp(-(1.0*i*i + 1.0*j*j) / (2.0 * s2));
				mat.ptr<float>(c + i)[c + j] = v;
				mat.ptr<float>(c - i)[c + j] = v;
				mat.ptr<float>(c + i)[c - j] = v;
				mat.ptr<float>(c - i)[c - j] = v;
			}
		}

		return mat;
	}

float ConvolveLocWidth(float* kernel, int dim, Mat ori, int x, int y)
	{
		unsigned int i;
		float pixel = 0;
		int col;
		int cen = dim / 2;
		for (i = 0; i < dim; i++)
		{
			col = x + (i - cen);
			if (col < 0) {
				col = 0;
				continue;
			}

			if (col >= ori.cols) {
				col = ori.cols - 1;
				continue;
			}
			pixel += kernel[i] * ori.ptr<float>(y)[col];;
		}
		if (pixel > 1)
			pixel = 1;
		return pixel;
	}

void Convolve1DWidth(float* kern, int dim, Mat ori, Mat dst)
	{
		unsigned int i, j;

		for (j = 0; j < ori.rows; j++)
		{
			for (i = 0; i < ori.cols; i++)
			{
				dst.ptr<float>(j)[i] = ConvolveLocWidth(kern, dim, ori, i, j);
			}
		}
	}

float ConvolveLocHeight(float* kernel, int dim, Mat ori, int x, int y)
	{
		unsigned int j;
		float pixel = 0;
		int cen = dim / 2;
		for (j = 0; j < dim; j++)
		{
			int row = y + (j - cen);
			if (row < 0)
				row = 0;
			if (row >= ori.rows)
				row = ori.rows - 1;
			pixel += kernel[j] * ori.ptr<float>(row)[x];;
		}
		if (pixel > 1)
			pixel = 1;
		return pixel;
	}
 
void Convolve1DHeight(float* kern, int dim, Mat ori, Mat dst)
	{
		unsigned int i, j;
		for (j = 0; j < ori.rows; j++)
		{
			for (i = 0; i < ori.cols; i++)
			{
				dst.ptr<float>(j)[i] = ConvolveLocHeight(kern, dim, ori, i, j);
			}
		}
	}

//卷积模糊 
int BlurImage(Mat ori, Mat dst, float sigma)
	{
		float* convkernel;
		int dim = (int)max(3.0, 2.0 * GAUSSKERN * sigma + 1.0);
		if (dim % 2 == 0)
			dim++;
		Mat tempMat = Mat(ori.rows, ori.cols, CV_32FC1);
		convkernel = GaussianKernel1D(sigma, dim);
		Convolve1DWidth(convkernel, dim, ori, tempMat);
		Convolve1DHeight(convkernel, dim, tempMat, dst);
		return dim;
	}






