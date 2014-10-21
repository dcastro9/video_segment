#include "Decomposition.h"
#include "Defs.h"
#include <highgui.h>

void Decomposition::findMinMax(const cv::Mat & img,float &min,float &max)
{
	float * ptr = (float *) img.data;
	float lmin, lmax;
	lmin = lmax = *ptr;

	int size = img.rows * img.cols;
	for (int i = 0 ; i < size ; ++i , ++ptr){
		if (*ptr<lmin)
			lmin=*ptr;
		if (*ptr>lmax)
			lmax=*ptr;
	}

	min = lmin;
	max = lmax;
}

// Reproject so each vector given by [p0 p1] is of length <=1
void Decomposition::Reproject(cv::Mat & p0, cv::Mat & p1)
{
	//cv::Mat reprojection(p0.rows,p0.cols,OPTFLOW_TYPE, cv::Scalar(0));
	float * p0Ptr = (float *) p0.data;
	float * p1Ptr = (float *) p1.data;
	float rep;
	int size = p0.rows * p0.cols;
	for (int i = 0 ; i < size ; ++i, ++p0Ptr, ++p1Ptr){
		rep = std::max(1.0f, std::sqrtf((*p0Ptr)*(*p0Ptr)+(*p1Ptr)*(*p1Ptr)));
		if (rep > 1){
			*p0Ptr = (*p0Ptr) / rep;
			*p1Ptr = (*p1Ptr) / rep;
		}
	}
}
// Rescale an image to be in the range [minv maxv]
void Decomposition::rescale(cv::Mat & img1,cv::Mat & img2,float minv,float maxv)
{
	float min1,max1,min2,max2,min,max;
	findMinMax(img1,min1,max1);
	findMinMax(img2,min2,max2);
	min=std::min(min1,min2);
	max=std::max(max1,max2);

	float * img1Ptr = (float *) img1.data;
	float * img2Ptr = (float *) img2.data;
	
	int size = img1.rows * img1.cols;
	for (int i = 0 ; i < size ; ++i, ++img1Ptr, ++img2Ptr){
		*img1Ptr = ((*img1Ptr-min)/(max-min))*(maxv-minv)+minv;
		*img2Ptr = ((*img2Ptr-min)/(max-min))*(maxv-minv)+minv;
	}
}
// Decompose the input IMAGE into structure and texture parts using the
// Rudin-Osher-Fatemi method. The final output is a linear combination 
// of the decomposed texture and the structure parts.

// in1 - first image (input)
// in2 - second image (input)
// texture1 - texture part for first image (ouput)
// texture2 - texture part for second image (ouput)
// structure1 - structure part for first image or NULL if not needed(ouput)
// structure2 - structure part for second image or NULL if not needed (ouput)
// Example usage:
// Assuming img1_32g and img2_32g grayscale images
//
// IplImage *texture1=cvCreateImage(cvSize(img1_32g->width,img1_32g->height),IPL_DEPTH_32F,1);
// IplImage *texture2=cvCreateImage(cvSize(img2_32g->width,img2_32g->height),IPL_DEPTH_32F,1);
// IplImage *structure1=cvCreateImage(cvSize(img1_32g->width,img1_32g->height),IPL_DEPTH_32F,1);
// IplImage *structure2=cvCreateImage(cvSize(img2_32g->width,img2_32g->height),IPL_DEPTH_32F,1);
// structure_texture_decomposition_rof(img1_32g,img1_32g,texture1,texture2,structure1,structure2,1.0f/8.0f,100,textureAlpha);
void Decomposition::structureTextureDecompositionRof(const cv::Mat& in1,const cv::Mat& in2,cv::Mat& texture1,cv::Mat& texture2, cv::Mat* structure1, cv::Mat* structure2,float theta,int nIters,float alp, bool display)
{	
	cv::Mat new_in1 = in1.clone();
	cv::Mat new_in2 = in2.clone();
	cv::Mat im[] = {new_in1,new_in2};
	 //Rescale the input image to [-1 1]
	rescale(new_in1,new_in2,-1,1);
	// Backup orginal images

	int type = new_in1.type();

	cv::Mat o1 = new_in1.clone();
	cv::Mat o2 = new_in2.clone();

	cv::Mat p1(new_in1.rows, new_in1.cols, type);
	cv::Mat p2(new_in1.rows, new_in1.cols, type);
	cv::Mat tmp(new_in1.rows, new_in1.cols, type);
	cv::Mat tmp1(new_in1.rows, new_in1.cols, type);
	cv::Mat tmp2(new_in1.rows, new_in1.cols, type);
	cv::Mat I_x(new_in1.rows, new_in1.cols, type);
	cv::Mat I_y(new_in1.rows, new_in1.cols, type);
	cv::Mat div_p(new_in1.rows, new_in1.cols, type);
	cv::Mat p[] = {p1, p2};
	
	// stepsize
	float delta = 1.0f/(4.0f*theta);

	cv::Mat ker1 = (cv::Mat_<float>(1,3) << -1, 1, 0);
	cv::Mat ker2 = (cv::Mat_<float>(1,2) << -1, 1);
	cv::Mat ker3 = (cv::Mat_<float>(2,1) << -1, 1);
	cv::Mat ker4 = (cv::Mat_<float>(3,1) << -1, 1, 0);

	for (int i = 0; i < 2; ++i)
	{
		// Initialize dual variable p to be 0
		p[0].setTo(cv::Scalar(0));
		p[1].setTo(cv::Scalar(0));
		for (int iter = 0; iter < nIters; ++iter)
		{
			// Compute divergence
			cv::filter2D(p[0],tmp1,type,ker1,cv::Point(-1,-1),0, cv::BORDER_CONSTANT);
			cv::filter2D(p[1],tmp2,type,ker4,cv::Point(-1,-1),0, cv::BORDER_CONSTANT);

			div_p = tmp1 + tmp2;
			div_p *= theta;
			im[i].copyTo(tmp);
			tmp += div_p;
			
			cv::filter2D(tmp,I_x,type,ker2,cv::Point(0,0),0, cv::BORDER_REPLICATE);
			cv::filter2D(tmp,I_y,type,ker3,cv::Point(0,0),0, cv::BORDER_REPLICATE);

			I_x *= delta;
			I_y *= delta;

			// Update dual variable
			p[0] += I_x;
			p[1] += I_y;

			// Reproject to |p| <= 1
			Reproject(p[0],p[1]);	
		}
		// compute divergence    
		cv::filter2D(p[0],tmp1,type,ker1,cv::Point(-1,-1),0, cv::BORDER_CONSTANT); // same as first
		cv::filter2D(p[1],tmp2,type,ker4,cv::Point(-1,-1),0, cv::BORDER_CONSTANT);

		div_p = tmp1 + tmp2;
		div_p *= theta;
		im[i] += div_p;		
	}
	if (structure1 && structure2)
	{
		im[0].copyTo(*structure1);
		im[1].copyTo(*structure2);
		rescale(*structure1,*structure2,0,1);
	}
	im[0] *= alp;		
	im[1] *= alp;	
	texture1 = o1 - im[0];	
	texture2 = o2 - im[1];
	
	if (display){
		cv::Mat texture3(texture1);
		cv::Mat texture4(texture2);

		rescale(texture3,texture4,0,1);
		cv::imshow("Texture 1", texture3);
		cv::imshow("Texture 2", texture4);
		cv::waitKey(1);
	}
	rescale(texture1,texture2,0,255);
}

