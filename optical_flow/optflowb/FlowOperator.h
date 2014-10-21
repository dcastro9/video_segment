#pragma once
#include "cv.h"
#include "flowUV.h"
#include "CRSSparseMat.h"
#include "OpticalFlowParams.h"
#include "FArray.h"

class FlowOperator{

public:
	FlowOperator(const int row, const int cols);
	void construct(flowUV& UV, const cv::Mat& Du, const cv::Mat& Dv, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat It, const OpticalFlowParams& params);

	CRSSparseMat& getA(){return _A;};
	FArray& getb(){return _b;};

	~FlowOperator(void);


private:
	CRSSparseMat _A;
	FArray _b;
	int _rows;
	int _cols;
	int _cells;
	cv::Mat FMxD;
	cv::Mat FMxd;
	cv::Mat FMyD;
	cv::Mat FMyd;

	static void bulidA (CRSSparseMat* const A,
						const int	cols,
						const int	rows,
						const float* const		duu, 
						const float* const 		dvv,
						const float* const		duv,
						const float* const 		UminorUpper,
						const float* const 		UminorLower,
						const float* const 		UouterUpper,
						const float* const 		UouterLower,
						const float* const 		VminorUpper,
						const float* const 		VminorLower,
						const float* const 		VouterUpper,
						const float* const 		VouterLower);

	static void buildM (CRSSparseMat& A,
						const int	cols,
						const int	rows,
						const float* const 		UDiag,
						const float* const 		UminorUpper,
						const float* const 		UminorLower,
						const float* const 		UouterUpper,
						const float* const 		UouterLower,
						const float* const 		VDiag,
						const float* const 		VminorUpper,
						const float* const 		VminorLower,
						const float* const 		VouterUpper,
						const float* const 		VouterLower);
};