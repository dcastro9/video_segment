#include <iostream>
#include "highgui.h" 
#include "cv.h"
#include "WeightedMedianFilter.h"
#include "UtilsDebug.h"
#include "Defs.h"

void WeightedMedianFilter::divergence(const cv::Mat & flowU,const cv::Mat & flowV, cv::Mat & dest)
{
	cv::Mat px(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	cv::Mat py(flowV.rows, flowV.cols, OPTFLOW_TYPE);

	// Calc the Gradient of flowU & flowV into px, py respectivly

	// Take forward differences on left and right edges of px
	if (flowU.cols > 1){
		float * pxPtr = (float *) px.data;
		float * flowUPtr1 = (float *) flowU.data;

		for (int i = 0 ; i < flowU.rows ; ++i){
			
			//float * pyPtr = py.ptr<float>(i,0);			// didn't work for some reason
			//float * flowVPtr1 = flowV.ptr<float>(i,0);
			//float * flowVPtr2 = flowV.ptr<float>(i,1);

			(*(pxPtr + (i * px.cols))) = (*(flowUPtr1 + (i * flowU.cols) + 1)) - (*(flowUPtr1 + (i * flowU.cols)));
		}

		pxPtr = (float *) px.data + (px.cols - 1);
		flowUPtr1 = (float *) flowU.data + (flowU.cols - 2);

		for (int i = 0 ; i < flowU.rows ; ++i){
			(*(pxPtr + (i * px.cols))) = (*(flowUPtr1 + (i * flowU.cols) + 1)) - (*(flowUPtr1 + (i * flowU.cols)));
		}
	}

	// Take centered differences on interior points of px
	if (flowU.cols > 2){
		float * pxPtr;
		float * flowUPtr1;

		for (int i = 1 ; i < (flowU.cols - 1) ; ++i){
			pxPtr = (float *) px.data + i;
			flowUPtr1 = (float *) flowU.data + i;

			for (int i = 0 ; i < flowU.rows ; ++i){
				(*(pxPtr + (i * px.cols))) = ((*(flowUPtr1 + (i * flowU.cols) + 1)) - (*(flowUPtr1 + (i * flowU.cols) - 1))) / 2;
			}
		}
	}

	// Take forward differences on left and right edges of py
	if (flowV.rows > 1){
		float * pyPtr = py.ptr<float>(0);
		const float * flowVPtr1 = flowV.ptr<const float>(0);
		const float * flowVPtr2 = flowV.ptr<const float>(1);

		for (int i = 0 ; i < flowV.cols ; ++i){
			(*(pyPtr + i)) = (*(flowVPtr2 + i)) - (*(flowVPtr1 + i));
		}

		pyPtr = py.ptr<float>(flowV.rows - 1);
		flowVPtr1 = flowV.ptr<float>(flowV.rows - 2);
		flowVPtr2 = flowV.ptr<float>(flowV.rows - 1);

		for (int i = 0 ; i < flowV.cols ; ++i){
			(*(pyPtr + i)) = (*(flowVPtr2 + i)) - (*(flowVPtr1 + i));
		}
	}

	// Take centered differences on interior points of py
	if (flowV.rows > 2){
		float * pyPtr;
		const float * flowVPtr1;
		const float * flowVPtr2;

		for (int i = 1 ; i < (flowV.rows - 1) ; ++i){
			pyPtr = py.ptr<float>(i);
			flowVPtr1 = flowV.ptr<const float>(i-1);
			flowVPtr2 = flowV.ptr<const float>(i+1);
			for (int j = 0 ; j < flowV.cols ; ++j){
				(*(pyPtr + j)) = ((*(flowVPtr2 + j)) - (*(flowVPtr1 + j))) / 2;
			}
		}
	}

	// finish clac gradient. Add px and py to divergence

	dest = px + py;
}

void WeightedMedianFilter::partialDeriv(const cv::Mat & flowU, const cv::Mat & flowV, const cv::Mat & image1, const cv::Mat & image2 , cv::Mat & It)
{
	// interpolation methoed = cubic

	float d[] = {1, -8, 0, 8, -1};	// derive filter
	cv::Mat h(1,5,OPTFLOW_TYPE,d);
	h = h/12;
	float b = 0.5;		// blending ratio

	int H = image1.rows;
	int W = image1.cols;

	cv::Mat x = flowU.clone();
	cv::Mat y = flowV.clone();

	float * xPtr = (float *) x.data;
	float * yPtr = (float *) y.data;

	// create meshgrid
	for (int i = 0 ; i < x.rows ; ++i){
		for (int j = 0 ; j < x.cols ; ++j){
			*(xPtr + (i * x.cols) + j) += (j + 1);
		}
	}


	for (int i = 0 ; i < y.rows ; ++i){
		for (int j = 0 ; j < y.cols ; ++j){
			*(yPtr + (i * y.cols) + j) += (i+1);
		}
	}

	// Record out of boundary pixels
	cv::Mat B1 = x.clone();
	cv::Mat B2 = y.clone();

	float * B1Ptr = (float *) B1.data;
	float * B2Ptr = (float *) B2.data;

	for (int i = 0 ; i < B1.rows * B1.cols ; ++i , ++B1Ptr, ++B2Ptr){
		*B1Ptr = ((*B1Ptr < 1) | (*B1Ptr > W));
		*B2Ptr = ((*B2Ptr < 1) | (*B2Ptr > H));
	}

	cv::Mat B = B1 | B2;

	cv::subtract(image2, image1,It);		// maybe this could be erased and just pass It from previous calculation

	float * ItPtr = (float *) It.data;
	float * BPtr = (float *) B.data;
	for (int i = 0 ; i < It.rows * It.cols ; ++i , ++ItPtr , ++BPtr){
		if (*BPtr == 1){
			*ItPtr = 0;
		}
	}
}

void WeightedMedianFilter::detect_occlusion(const cv::Mat & flowU,const cv::Mat & flowV,const  cv::Mat & image1,const  cv::Mat & image2, cv::Mat & occ)
{
	float sigma_d = 0.3f; // divergence
	float sigma_i = 20.0f;  // intensity

	// create Divregence destination
	cv::Mat div(flowU.rows, flowU.cols, OPTFLOW_TYPE);

	// init with 1
	occ.setTo(cv::Scalar(1));

	
	// Calc the Divergence of the flow, and put the answer in div
	divergence(flowU, flowV, div);

	// Go over the Div matrix and for each cell which is > 0 -> change to zero
	float * p = (float *) div.data;
	for (int i = 0 ; i < div.cols*div.rows ; ++i, ++p){
		if ((*p) > 0){
			(*p) = 0;
		}
	}

	cv::Mat It(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	partialDeriv(flowU, flowV, image1, image2, It);						// not tested

	// create temporary calcs of div for occlusion
	cv::Mat tmp1(div.rows, div.cols, OPTFLOW_TYPE);
	cv::exp((-1 * (div.mul(div)) / (sigma_d * sigma_d)), tmp1);			// not tested

	// create temporary calcs of it for occlusion
	cv::Mat tmp2(It.rows, It.cols, OPTFLOW_TYPE);
	cv::exp((((-1 * (It.mul(It))) / 2) / (sigma_i * sigma_i)), tmp2);	// not tested

	occ = tmp1.mul(tmp2);

}


void WeightedMedianFilter::findEdges(const cv::Mat & src, cv::Mat & dest, double threshold)
{
	//if (threshold == -1){
		float x[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
		float y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

		cv::Mat bx(src.rows, src.cols, OPTFLOW_TYPE);
		cv::Mat by(src.rows, src.cols, OPTFLOW_TYPE);
		cv::Mat b(src.rows, src.cols, OPTFLOW_TYPE);
		

		//x derivative
		cv::Mat kernelX(3, 3, OPTFLOW_TYPE, x);
		cv::Mat kernelY(3, 3, OPTFLOW_TYPE, y);
		kernelX /= 8;
		kernelY /= 8;

		cv::filter2D(src, bx, bx.depth(), kernelX,cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
		cv::filter2D(src, by, by.depth(), kernelY,cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
		b = bx.mul(bx) + by.mul(by);

		float bla = (float) cv::mean(b).val[0];

		float scale = 4;
		float cutoff = scale * bla;
		threshold = sqrt(cutoff);
	
	//}


/*	cv::Mat sobelX;
	cv::Mat sobelY;
	// Compute norm of Sobel
   cv::Sobel(src,sobelX,CV_8U,1,0);
   cv::Sobel(src,sobelY,CV_8U,0,1);
   cv::Mat sobel;
   //compute the L1 norm
   sobel= abs(sobelX)+abs(sobelY);

	// Find Sobel max value
   double sobmin, sobmax;
   cv::minMaxLoc(sobel,&sobmin,&sobmax);
   // Conversion to 8-bit image
   // sobelImage = -alpha*sobel + 255
   cv::Mat sobelImage;
   sobel.convertTo(sobelImage,CV_32F,-255./sobmax,255);

	cv::threshold(sobelImage, dest, threshold, 255, cv::THRESH_BINARY);	*/

	cv::threshold(b,dest,cutoff,1,cv::THRESH_BINARY);
}

void WeightedMedianFilter::medFilt(const cv::Mat & src, int filterSize ,cv::Mat & dest)
{
	//cv::Mat domain(filterSize,filterSize,OPTFLOW_TYPE, cv::Scalar(1));

	//int order = ((filterSize * filterSize) + 1)/2;

	// order filter
	int center = ((filterSize + 1) / 2);
	int padSize = filterSize - center;

	// create a Padded source Matrix into PaddedMat
	cv::Mat paddedMat(src.rows + (2 * padSize), src.cols + (2 * padSize), OPTFLOW_TYPE, cv::Scalar(0));
	cv::Mat paddedMat2(paddedMat, cv::Rect(padSize,padSize,src.cols,src.rows));
	src.copyTo(paddedMat2);
	
	// copy values into padded boundaries symetrically
	float* srcPtr;
	float* destPtr;

	
	// copy the first two rows
	for (int i = 0 ; i < padSize ; ++i){
		srcPtr = (float *) paddedMat.ptr(i + padSize);
		destPtr = (float *) paddedMat.ptr((padSize - 1) - i);

		for (int j = 0 ; j < paddedMat.cols; ++j, ++srcPtr , ++destPtr){
			*destPtr = *srcPtr;
		}
	}

	// copy the last two rows
	for (int i = 0 ; i < padSize ; ++i){
		srcPtr = (float *) paddedMat.ptr((paddedMat.rows) - (2 * padSize) + i);
		destPtr = (float *) paddedMat.ptr((paddedMat.rows - 1) - i);

		for (int j = 0 ; j < paddedMat.cols; ++j, ++srcPtr , ++destPtr){
			*destPtr = *srcPtr;
		}
	}

	// copy the first two cols
	for (int i = 0 ; i < padSize ; ++i){
		srcPtr = (float *) paddedMat.data + padSize + i;
		destPtr = (float *) paddedMat.data + ((padSize - 1) - i);

		for (int j = 0 ; j < paddedMat.rows; ++j){
			*(destPtr + (j * paddedMat.cols)) = *(srcPtr + (j * paddedMat.cols));
		}
	}

	// copy the last two cols
	for (int i = 0 ; i < padSize ; ++i){
		srcPtr = (float *) paddedMat.data + (paddedMat.cols - (2 * padSize)) + i;
		destPtr = (float *) paddedMat.data + (paddedMat.cols - 1) - i;

		for (int j = 0 ; j < paddedMat.rows; ++j){
			*(destPtr + (j * paddedMat.cols)) = *(srcPtr + (j * paddedMat.cols));
		}
	}

	// finished Padding
	
	// run median filter on padded matrix
	cv::Mat paddedMedian(paddedMat.rows,paddedMat.cols,OPTFLOW_TYPE);
	cv::medianBlur(paddedMat, paddedMedian, filterSize);

	// cut out into the dest matrix, the filtered matrix without the padding
	srcPtr = (float *) paddedMedian.data;
	destPtr = (float *) dest.data;
	
	for (int i = padSize ; i < (paddedMedian.rows - padSize) ; ++i){
		for (int j = padSize ; j < (paddedMedian.cols - padSize) ; ++j){
			*(destPtr + ((i - padSize) * dest.cols) + (j - padSize)) = *(srcPtr + (i * paddedMedian.cols) + j);
		}
	}

}


void WeightedMedianFilter::runMedianFilter(cv::Mat & flowU, cv::Mat &  flowV, const cv::Mat & image1, const cv::Mat & image2, int filterSize,int area_hsz, int sigma_i, cv::Mat & occ)
{
	/*float sigma_x = 7;
	int dilate = 5;
	*/
	cv::Mat flowUMedian(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	cv::Mat flowVMedian(flowV.rows, flowV.cols, OPTFLOW_TYPE);
	medFilt(flowU, filterSize, flowUMedian);
	medFilt(flowV, filterSize, flowVMedian);

	//cv::Mat edgesU(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	//cv::Mat edgesV(flowV.rows, flowV.cols, OPTFLOW_TYPE);
	//cv::Mat edges(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	//findEdges(flowU,edgesU);
	//findEdges(flowV,edgesV);

	//edges = edgesU | edgesV;

	// thin edges
	// dilate edges
	// compute weightes
	// run weighted filter
	
	flowUMedian.copyTo(flowU);
	flowVMedian.copyTo(flowV);
}


void WeightedMedianFilter::computeMedianFilter(cv::Mat & flowU, cv::Mat & flowV, const  cv::Mat & image1, const  cv::Mat & image2, int filterSize, int area_hsz, int sigma_i)
{
	//cv::Mat occ(flowU.rows, flowU.cols, OPTFLOW_TYPE);
	//detect_occlusion(flowU, flowV, image1, image2, occ);
	
	//runMedianFilter(flowU, flowV, image1, image2, filterSize, area_hsz,sigma_i,occ);
	runMedianFilter(flowU, flowV, image1, image2, filterSize, area_hsz,sigma_i,cv::Mat());
}
