#pragma once
#include"pre.h"

using namespace cv;

#define GridSpacing 4 
#define MAG(ROW,COL) ((float)((Mag).ptr<float>(ROW)[COL]) )   
#define ORI(ROW,COL) ((float)((Ori).ptr<float>(ROW)[COL]) )  
#define GRADX(ROW,COL) ((float)((gradx).ptr<float>(ROW)[COL])  ) 
#define GRADY(ROW,COL) ((float)((grady).ptr<float>(ROW)[COL])  )
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)(DOGoctaves[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL])

class sift {
public:
	int descrip_lenth = 0;
	int     nct = 0;
	ImageOctave *DOGoctaves;
	ImageOctave *mag_pyr; 
	ImageOctave *grad_pyr;
	Keypoint *keypoints = NULL; 
	Keypoint *keyDescriptors = NULL; 


	//---------------------------------------------------------------------------------------
	//1.
	Mat ScaleInitImage(Mat im)  //输入im为灰度图像32F
	{
		double preblur_sigma;
		Mat imMat = im.clone();
		int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN *INITSIGMA + 1.0);
		gaussdim = 2 * (gaussdim / 2) + 1;
		GaussianBlur(imMat, imMat, Size(gaussdim, gaussdim), INITSIGMA);
		if (DOUBLE_BASE_IMAGE_SIZE)
		{
			Mat bottom_Mat;
			resize(imMat, bottom_Mat, Size(imMat.rows * 2, imMat.cols * 2));
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN *preblur_sigma + 1.0);
			gaussdim = 2 * (gaussdim / 2) + 1;
			GaussianBlur(bottom_Mat, bottom_Mat, Size(gaussdim, gaussdim), preblur_sigma);
			return bottom_Mat;
		}
		else
		{
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN *preblur_sigma + 1.0);
			BlurImage(imMat, imMat, preblur_sigma); 
			return imMat;
		}
	}

	//2.  
	ImageOctave* BuildGaussianOctaves(Mat img)
	{
		ImageOctave *octaves;
		octaves = new ImageOctave[nct];
		DOGoctaves = new ImageOctave[nct];
		//计算阶梯内的层数
		int num_peroc_levels = SCALESPEROCTAVE + 3;
		int num_perdog_levels = num_peroc_levels - 1;
		Mat tempMat = img.clone(), dst, temp;
		float init_sigma = pow(2, 1.0 / 2);
		double k = pow(2, 1.0 / ((float)SCALESPEROCTAVE));

		for (int i = 0; i < nct; i++)
		{
			{
				//分配内存
				octaves[i].Octave = new ImageLevel[num_peroc_levels];
				DOGoctaves[i].Octave = new ImageLevel[num_perdog_levels];
				//首先建立金字塔每一阶梯的最底层，其中0阶梯的最底层已经建立好  
				(octaves[i].Octave)[0].Level = tempMat;
				octaves[i].col = tempMat.cols;
				octaves[i].row = tempMat.rows;
				DOGoctaves[i].col = tempMat.cols;
				DOGoctaves[i].row = tempMat.rows;
				if (DOUBLE_BASE_IMAGE_SIZE)
					octaves[i].subsample = pow(2, i)*0.5;
				else
					octaves[i].subsample = pow(2, i);
			}

			if (i == 0)
			{
				(octaves[0].Octave)[0].levelsigma = init_sigma;
				(octaves[0].Octave)[0].absolute_sigma = init_sigma / 2;
			}
			else
			{
				(octaves[i].Octave)[0].levelsigma = init_sigma;
				(octaves[i].Octave)[0].absolute_sigma = (octaves[i - 1].Octave)[num_peroc_levels - 3].absolute_sigma;
			}

			float sigma = init_sigma;
			float sigma_act, absolute_sigma;
			for (int j = 1; j < SCALESPEROCTAVE + 3; j++)
			{
				dst = Mat(tempMat.rows, tempMat.cols, CV_32FC1); 
				temp = Mat(tempMat.rows, tempMat.cols, CV_32FC1);

				sigma_act = sqrt(k*k - 1)*sigma;
				sigma = k * sigma;

				(octaves[i].Octave)[j].levelsigma = sigma;
				(octaves[i].Octave)[j].absolute_sigma = sigma * (octaves[i].subsample);
				int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN *sigma_act + 1.0);
				gaussdim = 2 * (gaussdim / 2) + 1;
				GaussianBlur((octaves[i].Octave)[j - 1].Level, dst, Size(gaussdim, gaussdim), sigma_act);
				(octaves[i].Octave)[j].levelsigmalength = gaussdim;
				(octaves[i].Octave)[j].Level = dst;

				temp = ((octaves[i].Octave)[j]).Level - ((octaves[i].Octave)[j - 1]).Level;
				((DOGoctaves[i].Octave)[j - 1]).Level = temp;
			}
			resize(((octaves[i].Octave)[SCALESPEROCTAVE].Level), tempMat, Size(((octaves[i].Octave)[SCALESPEROCTAVE].Level).rows / 2, ((octaves[i].Octave)[SCALESPEROCTAVE].Level).cols / 2));
		}
		return octaves;
	}
	//detect keypoint  
	int DKP(int nct, ImageOctave *GaussianPyr)
	{
		//计算用于DOG极值点检测的主曲率比的阈值  
		double curvature_threshold = ((CURVATURE_THRESHOLD + 1)*(CURVATURE_THRESHOLD + 1)) / CURVATURE_THRESHOLD;
		curvature_threshold = 10;
		int   keypoint_count = 0;
		for (int i = 0; i<nct; i++)
		{
			for (int j = 1; j<SCALESPEROCTAVE + 1; j++)
			{
				int dim = (int)(0.5*((GaussianPyr[i].Octave)[j].levelsigmalength) + 0.5);
				for (int m = dim; m<((DOGoctaves[i].row) - dim); m++)
					for (int n = dim; n<((DOGoctaves[i].col) - dim); n++)
					{
						if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
						{
							if (ImLevels(i, j, m, n) != 0.0) 
							{
								float inf_val = ImLevels(i, j, m, n);
								if (((inf_val <= ImLevels(i, j - 1, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m - 1, n)) &&
									(inf_val <= ImLevels(i, j - 1, m, n)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n)) &&
									(inf_val <= ImLevels(i, j - 1, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j - 1, m, n + 1)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n + 1)) &&   

									(inf_val <= ImLevels(i, j, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m, n - 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m - 1, n)) &&
									(inf_val <= ImLevels(i, j, m + 1, n)) &&
									(inf_val <= ImLevels(i, j, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j, m, n + 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n + 1)) &&    

									(inf_val <= ImLevels(i, j + 1, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m, n)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n + 1))    
									) ||
									((inf_val >= ImLevels(i, j - 1, m - 1, n - 1)) &&
									(inf_val >= ImLevels(i, j - 1, m, n - 1)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j - 1, m - 1, n)) &&
										(inf_val >= ImLevels(i, j - 1, m, n)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n)) &&
										(inf_val >= ImLevels(i, j - 1, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j - 1, m, n + 1)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n + 1)) &&

										(inf_val >= ImLevels(i, j, m - 1, n - 1)) &&
										(inf_val >= ImLevels(i, j, m, n - 1)) &&
										(inf_val >= ImLevels(i, j, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j, m - 1, n)) &&
										(inf_val >= ImLevels(i, j, m + 1, n)) &&
										(inf_val >= ImLevels(i, j, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j, m, n + 1)) &&
										(inf_val >= ImLevels(i, j, m + 1, n + 1)) &&

										(inf_val >= ImLevels(i, j + 1, m - 1, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m - 1, n)) &&
										(inf_val >= ImLevels(i, j + 1, m, n)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n)) &&
										(inf_val >= ImLevels(i, j + 1, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j + 1, m, n + 1)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n + 1))
										)) 
								{
									if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
									{
										float Dxx, Dyy, Dxy, Tr_H, Det_H, curvature_ratio;
										Dxx = ImLevels(i, j, m, n - 1) + ImLevels(i, j, m, n + 1) - 2.0*ImLevels(i, j, m, n);
										Dyy = ImLevels(i, j, m - 1, n) + ImLevels(i, j, m + 1, n) - 2.0*ImLevels(i, j, m, n);
										Dxy = ImLevels(i, j, m - 1, n - 1) + ImLevels(i, j, m + 1, n + 1) - ImLevels(i, j, m + 1, n - 1) - ImLevels(i, j, m - 1, n + 1);
										Tr_H = Dxx + Dyy;
										Det_H = Dxx * Dyy - Dxy * Dxy;
										// Compute the ratio of the principal curvatures.  
										curvature_ratio = (1.0*Tr_H*Tr_H) / Det_H;
										if ((Det_H >= 0.0) && (curvature_ratio <= curvature_threshold))  //最后得到最具有显著性特征的特征点  
										{
											keypoint_count++;
											Keypoint *k;
											k = (Keypoint*)malloc(sizeof(struct Keypoint));
											k->next = keypoints;
											keypoints = k;
											k->row = m * (GaussianPyr[i].subsample);
											k->col = n * (GaussianPyr[i].subsample);
											k->sy = m;    //行  
											k->sx = n;    //列  
											k->octave = i;
											k->level = j;
											k->scale = (GaussianPyr[i].Octave)[j].absolute_sigma;
										}//if >curvature_thresh  
									}//if >contrast  
								}//if inf value  
							}//if non zero  
						}//if >contrast  
					}  //for concrete image level col  
			}//for levels  
		}//for octaves  
		return keypoint_count;
	}
	//detect keypoint location 
	void DKPL(Mat image, ImageOctave *GaussianPyr)
	{

		Keypoint *p = keypoints;
		while (p) 
		{
			circle(image, Point2i(int(p->row), int(p->col)), 3, Scalar(0, 255, 255));
			p = p->next;
		}

	}
	
	//4.
	void ComputeGrad_DirecandMag(ImageOctave *GaussianPyr)
	{
		mag_pyr = new ImageOctave[nct];
		grad_pyr = new ImageOctave[nct];
//#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)(GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL])
		for (int i = 0; i<nct; i++)
		{
			mag_pyr[i].Octave = new ImageLevel[SCALESPEROCTAVE];
			grad_pyr[i].Octave = new ImageLevel[SCALESPEROCTAVE];
			for (int j = 1; j<SCALESPEROCTAVE + 1; j++)
			{
				Mat Mag(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat Ori(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat gradx(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat grady(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				for (int m = 1; m<(GaussianPyr[i].row - 1); m++)
					for (int n = 1; n<(GaussianPyr[i].col - 1); n++)
					{
						//计算幅值  
						(gradx).ptr<float>(m)[n] = 0.5*(ImLevels(i, j, m, n + 1) - ImLevels(i, j, m, n - 1));  //dx  
						(grady).ptr<float>(m)[n] = 0.5*(ImLevels(i, j, m + 1, n) - ImLevels(i, j, m - 1, n));  //dy  
						(Mag).ptr<float>(m)[n] = sqrt(GRADX(m, n)*GRADX(m, n) + GRADY(m, n)*GRADY(m, n));  //mag  

																										   //atan的范围是 （-PI/2，PI/2 ）
						(Ori).ptr<float>(m)[n] = atan(GRADY(m, n) / GRADX(m, n)); //+ ((gradx).ptr<float>(m)[n] < 0 ? CV_PI : 0);
						if ((gradx).ptr<float>(m)[n] < 0) {
							(Ori).ptr<float>(m)[n] = (Ori).ptr<float>(m)[n] + CV_PI;
						}
						if (ORI(m, n) >= CV_PI)
							(Ori).ptr<float>(m)[n] = (Ori).ptr<float>(m)[n] - 2 * CV_PI;
					}
				((mag_pyr[i].Octave)[j - 1]).Level = Mag;
				((grad_pyr[i].Octave)[j - 1]).Level = Ori;
			}//for levels  
		}//for octaves  
	}
	int FindClosestRotationBin(int binCount, float angle)
	{
		angle += CV_PI;
		angle /= 2.0 * CV_PI;
		angle *= binCount;
		int idx = (int)angle;
		if (idx == binCount)
			idx = 0;
		return (idx);
	}
	void AverageWeakBins(double* hist, int binCount)
	{
		for (int sn = 0; sn < 2; ++sn)
		{
			double firstE = hist[0];
			double last = hist[binCount - 1];
			for (int sw = 0; sw < binCount; ++sw)
			{
				double cur = hist[sw];
				double next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];
				hist[sw] = (last + cur + next) / 3.0;
				last = cur;
			}
		}
	}
	bool InterpolateOrientation(double left, double middle, double right, double *degreeCorrection, double *peakValue)
	{
		double a = ((left + right) - 2.0 * middle) / 2.0;   
		if (a == 0.0)
			return false;
		double c = (((left - middle) / a) - 1.0) / 2.0;
		double b = middle - c * c * a;
		if (c < -0.5 || c > 0.5)
			return false;
		*degreeCorrection = c;
		*peakValue = b;
		return true;
	}
	void AssignTheMainOrientation(int nct, ImageOctave *GaussianPyr, ImageOctave *mag_pyr, ImageOctave *grad_pyr)
	{
		int num_bins = 36;
		float hist_step = (2.0*CV_PI) / (num_bins + 0.0);
		float hist_orient[36];
		for (int i = 0; i < 36; i++) {
			hist_orient[i] = -CV_PI + i * hist_step;
		}

		float sigma1 = (((GaussianPyr[0].Octave)[SCALESPEROCTAVE].absolute_sigma)) / (GaussianPyr[0].subsample);
		int zero_pad = (int)((int(std::max(3.0, 2 * GAUSSKERN *sigma1 + 1.0))) / 2 + 1);
		int keypoint_count = 0;
		Keypoint *p = keypoints;
		double* orienthist = new double[36];


		while (p)
		{
			int i = p->octave;
			int j = p->level;
			int m = p->sy;
			int n = p->sx; 
			if ((m >= zero_pad) && (m<GaussianPyr[i].row - zero_pad) &&
				(n >= zero_pad) && (n<GaussianPyr[i].col - zero_pad))
			{
				float sigma = (((GaussianPyr[i].Octave)[j].absolute_sigma)) / (GaussianPyr[i].subsample);
				//产生二维高斯模板  
				Mat mat = GaussianKernel2D(sigma);
				int dim = (int)max(3.0, 2.0 * GAUSSKERN *sigma + 1.0);   dim = 2 * (dim / 2) + 1;
				dim = dim / 2;
				//分配用于存储Patch幅值和方向的空间  
#define MAT(ROW,COL) ((float)(mat).ptr<float>(ROW)[COL])
				for (int sw = 0; sw < 36; ++sw)
				{
					orienthist[sw] = 0.0;
				}
				for (int x = m - dim, mm = 0; x <= (m + dim); x++, mm++)
					for (int y = n - dim, nn = 0; y <= (n + dim); y++, nn++)
					{

						double mag = ((mag_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						double Ori = ((grad_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						int binIdx = FindClosestRotationBin(36, Ori);                   //得到离现有方向最近的直方块  
						orienthist[binIdx] = orienthist[binIdx] + 1.0* mag * MAT(mm, nn);//利用高斯加权累加进直方图相应的块  
					}
				// Find peaks in the orientation histogram using nonmax suppression.  
				AverageWeakBins(orienthist, 36);
				double maxGrad = 0.0;   //最大方向的权值
				int maxBin = 0;
				for (int b = 0; b < 36; ++b)
				{
					if (orienthist[b] > maxGrad)
					{
						maxGrad = orienthist[b];
						maxBin = b;
					}
				}

				double maxPeakValue = 0.0;
				double maxDegreeCorrection = 0.0;
				if ((InterpolateOrientation(orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],
					orienthist[maxBin],
					orienthist[(maxBin + 1) % 36],
					&maxDegreeCorrection,
					&maxPeakValue)) == false)
				{
					printf("BUG: Parabola fitting broken");
				}
				bool binIsKeypoint[36];
				for (int b = 0; b < 36; ++b)
				{
					binIsKeypoint[b] = false;

					if (b == maxBin)
					{
						binIsKeypoint[b] = true;
						continue;
					}
					if (orienthist[b] < (peakRelThresh * maxPeakValue))
						continue;

					int leftI = (b == 0) ? (36 - 1) : (b - 1);
					int rightI = (b + 1) % 36;
					if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])
						continue; 
					binIsKeypoint[b] = true;
				}

				double oneBinRad = (2.0 * CV_PI) / 36;
				for (int b = 0; b < 36; ++b)
				{
					if (binIsKeypoint[b] == true) {
						int bLeft = (b == 0) ? (36 - 1) : (b - 1);
						int bRight = (b + 1) % 36;

						double peakValue;
						double degreeCorrection;

						if (InterpolateOrientation(orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],
							orienthist[maxBin], orienthist[(maxBin + 1) % 36],
							&degreeCorrection, &peakValue) == false)
						{
							printf("BUG: Parabola fitting broken");
						}

						double degree = (b + degreeCorrection) * oneBinRad - CV_PI;
						if (degree < -CV_PI)
							degree += 2.0 * CV_PI;
						else if (degree > CV_PI)
							degree -= 2.0 * CV_PI;

						Keypoint *k = new Keypoint();
						k->next = keyDescriptors;
						keyDescriptors = k;
						k->row = p->row;
						k->col = p->col;
						k->sy = p->sy;     
						k->sx = p->sx;     
						k->octave = p->octave;
						k->level = p->level;
						k->scale = p->scale;

						k->ori = degree;
						k->mag = peakValue;
					}
				}
			}
			p = p->next;
		}
	}
	void DisplayOrientation(Mat image, ImageOctave *GaussianPyr)
	{
		Keypoint *p = keyDescriptors;
		while (p) 
		{
			float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
			float autoscale = 3.0;
			float uu = autoscale * scale*cos(p->ori);
			float vv = autoscale * scale*sin(p->ori);
			float x = (p->col) + uu;
			float y = (p->row) + vv;

			line(image, cvPoint((int)(p->col), (int)(p->row)),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);

			float alpha = 0.33;
			float beta = 0.6; 

			float xx0 = (p->col) + uu - alpha * (uu + beta * vv);
			float yy0 = (p->row) + vv - alpha * (vv - beta * uu);
			float xx1 = (p->col) + uu - alpha * (uu - beta * vv);
			float yy1 = (p->row) + vv - alpha * (vv + beta * uu);
			line(image, cvPoint((int)xx0, (int)yy0),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);
			line(image, cvPoint((int)xx1, (int)yy1),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);
			p = p->next;
		}
		imshow("oritation", image);
	}
		
	//5.
	void ExtractFeatureDescriptors(ImageOctave *GaussianPyr)
	{
		float feat_window = 2 * GridSpacing;
		float orient_bin_spacing = CV_PI / 4;
		float orient_angles[8] = { -CV_PI,-CV_PI + orient_bin_spacing,-CV_PI * 0.5, -orient_bin_spacing,
			0.0, orient_bin_spacing, CV_PI*0.5,  CV_PI + orient_bin_spacing };

		float *feat_grid = (float *)malloc(2 * 16 * sizeof(float));
		for (int i = 0; i<GridSpacing; i++)
		{
			for (int j = 0; j<2 * GridSpacing; j += 2)
			{
				feat_grid[i * 2 * GridSpacing + j] = -6.0 + i * GridSpacing;
				feat_grid[i * 2 * GridSpacing + j + 1] = -6.0 + 0.5*j*GridSpacing;
			}
		}

		float *feat_samples = (float *)malloc(2 * 256 * sizeof(float));
		for (int i = 0; i<4 * GridSpacing; i++)
		{
			for (int j = 0; j<8 * GridSpacing; j += 2)
			{
				feat_samples[i * 8 * GridSpacing + j] = -(2 * GridSpacing - 0.5) + i;
				feat_samples[i * 8 * GridSpacing + j + 1] = -(2 * GridSpacing - 0.5) + 0.5*j;
			}
		}
		Keypoint *p = keyDescriptors;
		while (p)
		{
			float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
			float sine = -sin(p->ori);
			float cosine = cos(p->ori);
			float *featcenter = (float *)malloc(2 * 16 * sizeof(float));
			for (int i = 0; i<GridSpacing; i++)
			{
				for (int j = 0; j<2 * GridSpacing; j += 2)
				{
					float x = feat_grid[i * 2 * GridSpacing + j];
					float y = feat_grid[i * 2 * GridSpacing + j + 1];
					featcenter[i * 2 * GridSpacing + j] = ((cosine * x + sine * y) + p->sx);
					featcenter[i * 2 * GridSpacing + j + 1] = ((-sine * x + cosine * y) + p->sy);
				}
			}

			float *feat = (float *)malloc(2 * 256 * sizeof(float));
			for (int i = 0; i<64 * GridSpacing; i++, i++)
			{
				float x = feat_samples[i];
				float y = feat_samples[i + 1];
				feat[i] = ((cosine * x + sine * y) + p->sx);
				feat[i + 1] = ((-sine * x + cosine * y) + p->sy);
			}
			float *feat_desc = (float *)malloc(128 * sizeof(float));
			for (int i = 0; i<128; i++)
			{
				feat_desc[i] = 0.0;
			}
			for (int i = 0; i<512; i += 2)
			{
				float mag_sample = 0, grad_sample = 0, x_sample = 0, y_sample = 0; {
					x_sample = feat[i];
					y_sample = feat[i + 1];
					float sample12 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample - 1);
					float sample21 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample - 1, y_sample);
					float sample22 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample);
					float sample23 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample + 1, y_sample);
					float sample32 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample + 1);
					float diff_x = sample23 - sample21;
					float diff_y = sample32 - sample12;
					mag_sample = sqrt(diff_x*diff_x + diff_y * diff_y);
					grad_sample = atan(diff_y / diff_x);
					if (diff_x < 0)
						grad_sample += CV_PI;
					if (grad_sample >= CV_PI)   grad_sample -= 2 * CV_PI;
				}

				float *pos_wght = (float *)malloc(8 * GridSpacing * GridSpacing * sizeof(float)); {
					float *x_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float));
					float *y_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float));
					for (int m = 0; m<32; ++m, ++m)
					{

						float x = featcenter[m];
						float y = featcenter[m + 1];
						x_wght[m / 2] = max(1 - (fabs(x - x_sample)*1.0 / GridSpacing), 0.0);
						y_wght[m / 2] = max(1 - (fabs(y - y_sample)*1.0 / GridSpacing), 0.0);
					}
					for (int m = 0; m<16; ++m)
						for (int n = 0; n<8; ++n)
							pos_wght[m * 8 + n] = x_wght[m] * y_wght[m];
					free(x_wght);
					free(y_wght);
				}
				float diff[8];
				for (int m = 0; m<8; ++m)
				{
					float angle = grad_sample - (p->ori) - orient_angles[m] + CV_PI; //差值+pi
					float temp = angle / (2.0 * CV_PI);
					angle -= (int)(temp) * (2.0 * CV_PI);
					diff[m] = angle - CV_PI;
				}
				float x = p->sx, y = p->sy;                                                                    //feat_window=2*GridSpacing=8  .原代码是不是少写了sqrt
				float g = exp(-((x_sample - x)*(x_sample - x) + (y_sample - y)*(y_sample - y)) / (2 * feat_window*feat_window)) / sqrt(2 * CV_PI*feat_window*feat_window);


				float orient_wght[128];
				for (int m = 0; m<128; ++m)
				{
					orient_wght[m] = max((1.0 - (1.0*fabs(diff[m % 8])) / orient_bin_spacing), 0.0);
					feat_desc[m] = feat_desc[m] + orient_wght[m] * pos_wght[m] * g*mag_sample;
				}
				free(pos_wght);
			}
			free(feat);
			free(featcenter);
			float norm = GetVecNorm(feat_desc, 128);
			for (int m = 0; m<128; m++)
			{
				feat_desc[m] /= norm;
				if (feat_desc[m]>0.2)
					feat_desc[m] = 0.2;
			}
			norm = GetVecNorm(feat_desc, 128);
			for (int m = 0; m<128; m++)
			{
				feat_desc[m] /= norm;
			}
			p->descrip = feat_desc;
			p = p->next;
			descrip_lenth++;
		}
		free(feat_grid);
		free(feat_samples);
	}


	//-----------------------------------------------------------------------------




	void release() {
		free(DOGoctaves);
		free(mag_pyr);
		free(grad_pyr);
		free(keypoints);
		free(keyDescriptors);
	}

};

