#include "FlowError.h"
#include "FlowUtils.h"
#include "UtilsMat.h"
#include "UtilsDebug.h"
#include "Defs.h"
#include <math.h>
#include <highgui.h>

float* FlowError::calcError(flowUV& UV, flowUV& GT,  bool display){
	int size = 0;
	cv::Mat mask(UV.getU().rows, UV.getU().cols, CV_8U,cv::Scalar(0));
	float* gu = (float*)GT.getU().data;
	float* gv = (float*)GT.getV().data;
	uchar* m =  mask.data;
	for (int i = 0 ; i < GT.getU().rows * GT.getU().cols; ++i, ++gu, ++gv, ++m){
		if ((std::abs(*gv) == 0 && std::abs(*gu) == 0) | (std::abs(*gv) > 1000000000) | (std::abs(*gu) > 1000000000))
		{
			*m = 0;
		}
		else
		{
			*m = 1;
			++size;
		}
	}

	float* u = (float*)UV.getU().data;
	float* v = (float*)UV.getV().data;
	gu = (float*)GT.getU().data;
	gv = (float*)GT.getV().data;
	cv::Mat msu(size, 1, OPTFLOW_TYPE);
	cv::Mat msv(size, 1, OPTFLOW_TYPE);
	cv::Mat mstu(size, 1, OPTFLOW_TYPE);
	cv::Mat mstv(size, 1, OPTFLOW_TYPE);
	float* pmsu = (float*)msu.data;
	float* pmsv = (float*)msv.data;
	float* pmstu = (float*)mstu.data;
	float* pmstv = (float*)mstv.data;
	m =  mask.data;
	for(int i = 0; i < UV.getU().rows * UV.getU().cols; ++i, ++u, ++v, ++gu, ++gv, ++m){
		if (*m){
			*(pmsu++) = *(u);
			*(pmsv++) = *(v);
			*(pmstu++) = *(gu);
			*(pmstv++) = *(gv);
		}
	}


	cv::Mat n(size, 1, OPTFLOW_TYPE,cv::Scalar(0));
	pmsu = (float*)msu.data;
	pmsv = (float*)msv.data;
	float* pn =  (float*)n.data;
	for(int i = 0; i < size; ++i, ++pmsu, ++pmsv, ++pn){
		*pn = (1.0f/ std::sqrtf((*pmsu) * (*pmsu) + (*pmsv) * (*pmsv) + 1));
	}

	cv::Mat un = msu.mul(n);
	cv::Mat vn = msv.mul(n);

	cv::Mat tn(size, 1, OPTFLOW_TYPE,cv::Scalar(0));
	pmstu = (float*)mstu.data;
	pmstv = (float*)mstv.data;
	float* ptn =  (float*)tn.data;
	for(int i = 0; i < size; ++i, ++pmstu, ++pmstv, ++ptn){
		*ptn = (1.0f/ std::sqrtf((*pmstu) * (*pmstu) + (*pmstv) * (*pmstv) + 1));
	}

	cv::Mat tun = mstu.mul(tn);
	cv::Mat tvn = mstv.mul(tn);
		
	cv::Mat ang(size, 1, OPTFLOW_TYPE,cv::Scalar(0));
	cv::Mat angtmp = un.mul(tun)+ vn.mul(tvn) +(n.mul(tn));
	float* pAng = (float*)ang.data;
	float* pAngtmp = (float*)angtmp.data;
	ptn = (float*)tn.data;
	for(int i = 0; i < size; ++i){
		*(pAng++) = std::acos(std::max<float>(-1, std::min<float>(1,*(pAngtmp++))));
	}

	float mang = (float)cv::mean(ang).val[0];
	static float pi = 3.14159265358979323846f;
	mang = mang * 180 / pi;

	cv::Scalar meanTmp, meanStd;
	cv::meanStdDev(cv::Mat(ang*180/pi), meanTmp, meanStd);
	float stdang = (float)meanStd.val[0];



	cv::Mat epe1 = (GT.getU() - UV.getU()).mul((GT.getU() - UV.getU())) + (GT.getV() - UV.getV()).mul((GT.getV() - UV.getV()));
	cv::Mat epe2(UV.getU().rows, UV.getU().cols, OPTFLOW_TYPE,cv::Scalar(0));;
	cv::sqrt(epe1, epe2);
	cv::Mat epe(size, 1, OPTFLOW_TYPE,cv::Scalar(0));
	float* pepe2 = (float*)epe2.data;
	float* pepe =  (float*)epe.data;
	m =  mask.data;
	for(int i = 0; i < UV.getU().rows * UV.getU().cols; ++i, ++pepe2, ++m){
		if (*m){
			*(pepe++) = *(pepe2);
		}
	} 
	float mepe = (float)cv::mean(epe).val[0];


	float* ret = new float[3];
	ret[0] = mang;
	ret[1] = stdang;
	ret[2] = mepe;
	if (display){
		UtilsMat::clamp<float>(epe2,0,1);
		pepe2 = (float*)epe2.data;
		m = mask.data;
		
		for(int i = 0; i < epe2.cols * epe2.rows; ++i, ++pepe2, ++m)
			*(pepe2) = *(pepe2) * *(m);

		cv::imshow("End Point Error Visualization", epe2);
		cv::waitKey(1);
	}
	return ret;
}
