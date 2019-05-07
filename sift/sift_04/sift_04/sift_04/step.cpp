#include"step.h"

void process(Res r) 
{
		Mat ori, grey, image, DoubleSizeImage;  
		Mat init_Mat, bottom_Mat;
		int rows, cols;
		
		sift *sift = r.sift;
		
		//pre-----------------------------------------------------------------------
		ori = r.img.clone();
		if (ori.rows == 0) {
			printf("no pic");
		}
		cvtColor(ori, grey, CV_BGR2GRAY);
		image = ori.clone();
		DoubleSizeImage = Mat(2 * ori.rows, 2 * ori.cols, CV_32FC3);
		init_Mat = Mat(ori.rows, ori.cols, CV_32FC1);
		grey.convertTo(init_Mat, CV_32FC1);
		cv::normalize(init_Mat, init_Mat, 1.0, 0.0, NORM_MINMAX);
		int nct = 4;
		int dim = min(init_Mat.rows, init_Mat.cols);
		nct = (int)(log((double)dim) / log(2.0)) - 2;
		nct = min(nct, MAXOCTAVES);
		sift->nct = nct;
		//-------------------------------------------------------------------------------
		
		//1.
		bottom_Mat = sift->ScaleInitImage(init_Mat);
		
		
		//2.
		ImageOctave *Gaussianpyr;
		Gaussianpyr = sift->BuildGaussianOctaves(bottom_Mat);
		
		
		//3. 
		int keycount = sift->DKP(nct, Gaussianpyr);
		sift->DKPL(r.img, Gaussianpyr);
		image.convertTo(image, CV_32FC3);
		cv::normalize(image, image, 1.0, 0.0, NORM_MINMAX);

		//4. 
		sift->ComputeGrad_DirecandMag(Gaussianpyr);
		sift->AssignTheMainOrientation(nct, Gaussianpyr, sift->mag_pyr, sift->grad_pyr);
		image = ori.clone();
		sift->DisplayOrientation(image, Gaussianpyr);


		//5. 
		sift->ExtractFeatureDescriptors(Gaussianpyr);


		//ÊÍ·ÅÄÚ´æ
		ori.release();
		grey.release();
		image.release();
		DoubleSizeImage.release();
		init_Mat.release();
		bottom_Mat.release();
		free(Gaussianpyr);

		printf("--------------------------------------------------\n");
}