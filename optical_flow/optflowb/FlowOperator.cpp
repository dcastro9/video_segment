#include "FlowOperator.h"
#include "Defs.h"
#include "PenaltyFunctionCompute.h"
#include "UtilsMat.h"

FlowOperator::FlowOperator(const int rows, const int cols) : _b(2 * rows * cols, true, 0), _rows(rows), _cols(cols), _cells(rows * cols), FMxD(rows, cols, OPTFLOW_TYPE), FMxd(rows, cols, OPTFLOW_TYPE), FMyD(rows, cols, OPTFLOW_TYPE), FMyd(rows, cols, OPTFLOW_TYPE){
	//construct consts
		//FMx/Fx const
	//FMx contains the FMxD mat on its main diagonal and FMxd on the -cols(in matlab its -rows) diagonal
	FArray zeros(_cols, true, 0);
	FArray ones(_cols, true, 1);
	FArray nones(_cols, true, -1);

	/*
	0  1  1  1       0  0  0  0
	0  1  1  1  ->   1  1  1  1
	0  1  1  1       1  1  1  1
	0  1  1  1       1  1  1  1
	*/
	//cv::Mat FMxD(_rows, _cols, OPTFLOW_TYPE);
	memcpy(FMxD.ptr<float>(0), zeros.ptr, _cols * sizeof(float));
	for(int i = 1; i < _rows; ++i)
		memcpy(FMxD.ptr<float>(i), ones.ptr, _cols * sizeof(float));
	/*
	-1 -1 -1  0      -1 -1 -1 -1
	-1 -1 -1  0  ->  -1 -1 -1 -1
	-1 -1 -1  0      -1 -1 -1 -1
	-1 -1 -1  0       0  0  0  0
	*/
	//cv::Mat FMxd(_rows, _cols, OPTFLOW_TYPE, cv::Scalar(-1)); increases quality
	//cv::Mat FMxd(_rows, _cols, OPTFLOW_TYPE);
	for(int i = 1; i < _rows; ++i)
		memcpy(FMxd.ptr<float>(i), nones.ptr, _cols * sizeof(float));
	memcpy(FMxd.ptr<float>(0), zeros.ptr, _cols * sizeof(float));


	//FMy/Fy const

	/*
	 0  0  0  0       0  1  1  1
	 1  1  1  1  ->   0  1  1  1
	 1  1  1  1       0  1  1  1
	 1  1  1  1       0  1  1  1
	*/
	FArray line1(_cols, true, 1);
	*(line1.ptr) = 0;	
	//cv::Mat FMyD(_rows, _cols, OPTFLOW_TYPE);
	for(int i = 0; i < _rows; ++i)
		memcpy(FMyD.ptr<float>(i), line1.ptr, _cols * sizeof(float));
	
	/*
	-1 -1 -1 -1       -1 -1 -1  0
	-1 -1 -1 -1   ->  -1 -1 -1  0
	-1 -1 -1 -1       -1 -1 -1  0
	 0  0  0  0       -1 -1 -1  0
	*/
	FArray line2(_cols, true, -1);
	*(line2.ptr + _cols - 1) = 0;
	//cv::Mat FMyd(_rows, _cols, OPTFLOW_TYPE);
	for(int i = 0; i < _rows; ++i)
		memcpy(FMyd.ptr<float>(i), line2.ptr, _cols * sizeof(float));
}

FlowOperator::~FlowOperator(){
}



void FlowOperator::bulidA (CRSSparseMat* const A,
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
						const float* const 		VouterLower)
{

	int majorDiagSize = rows * cols;
	int minorDiagSize = (rows * cols) - 1;
	int outerDiagSize = (rows - 1) * cols;
	int nzAestimate = 4 * (majorDiagSize + minorDiagSize + outerDiagSize);

	float* cooVal = new float[nzAestimate];
	float* cooValPtr = cooVal;

	memcpy((cooValPtr),						UouterLower,	outerDiagSize * sizeof(float));
	memcpy((cooValPtr += outerDiagSize),	UminorLower,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	duu,		majorDiagSize * sizeof(float));
	memcpy((cooValPtr += majorDiagSize),	UminorUpper,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	UouterUpper,	outerDiagSize * sizeof(float));
	memcpy((cooValPtr += outerDiagSize),	duv,		majorDiagSize * sizeof(float));

	memcpy((cooValPtr += majorDiagSize),	duv,		majorDiagSize * sizeof(float));
	memcpy((cooValPtr += majorDiagSize),	VouterLower,	outerDiagSize * sizeof(float));
	memcpy((cooValPtr += outerDiagSize),	VminorLower,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	dvv,		majorDiagSize * sizeof(float));
	memcpy((cooValPtr += majorDiagSize),	VminorUpper,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	VouterUpper,	outerDiagSize * sizeof(float));

	int* cooRow = new int[nzAestimate];
	int* cooCol = new int[nzAestimate];
	int* cooRowPtr = cooRow;
	int* cooColPtr= cooCol;

	//TODO:: //BOAZ: maybe the count forvar (and the other redundent forvars) forvar has a relative big overhead.. needs to be benchmarked
	for (int c = 0,				r = cols;	c < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Outer
	for (int c = 0,				r = 1;		c < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Inner
	for (int c = 0,				r = 0;		c < majorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//duu
	for (int c = 1,				r = 0;		r < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Inner
	for (int c = cols,			r = 0;		r < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Outer
	for (int c = majorDiagSize,	r = 0;		r < majorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//duv
	
	for (int c = 0,						r = majorDiagSize;			c < majorDiagSize; *(cooRowPtr++) = r,	*(cooColPtr++) = c, ++c, ++r);							//duv
	for (int c = majorDiagSize,			r = majorDiagSize + cols,	count = 0;	count < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Outer
	for (int c = majorDiagSize,			r = majorDiagSize + 1,		count = 0;	count < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Inner
	for (int c = majorDiagSize,			r = majorDiagSize,			count = 0;	count < majorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//dvv
	for (int c = majorDiagSize + 1,		r = majorDiagSize,			count = 0;	count < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Inner
	for (int c = majorDiagSize + cols,	r = majorDiagSize,			count = 0;	count < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Outer

	COOrdSparseMat* coo = new COOrdSparseMat(2 * majorDiagSize, 2 * majorDiagSize, nzAestimate, cooVal, cooRow, cooCol);

	A->build(*coo);

	delete coo;

}

void FlowOperator::buildM (CRSSparseMat& const A,
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
						const float* const 		VouterLower)
{

	int majorDiagSize = rows * cols;
	int minorDiagSize = (rows * cols) - 1;
	int outerDiagSize = (rows - 1) * cols;
	int nzAestimate = 2 * majorDiagSize + 4 * minorDiagSize + 4 * outerDiagSize;

	float* cooVal = new float[nzAestimate];
	float* cooValPtr = cooVal;

	memcpy((cooValPtr),						UouterLower,	outerDiagSize * sizeof(float));
	memcpy((cooValPtr += outerDiagSize),	UminorLower,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	UDiag,		majorDiagSize * sizeof(float));
	memcpy((cooValPtr += majorDiagSize),	UminorUpper,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	UouterUpper,	outerDiagSize * sizeof(float));

	memcpy((cooValPtr += outerDiagSize),	VouterLower,	outerDiagSize * sizeof(float));
	memcpy((cooValPtr += outerDiagSize),	VminorLower,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	VDiag,		majorDiagSize * sizeof(float));
	memcpy((cooValPtr += majorDiagSize),	VminorUpper,	minorDiagSize * sizeof(float));
	memcpy((cooValPtr += minorDiagSize),	VouterUpper,	outerDiagSize * sizeof(float));

	int* cooRow = new int[nzAestimate];
	int* cooCol = new int[nzAestimate];
	int* cooRowPtr = cooRow;
	int* cooColPtr= cooCol;

	//TODO:: //BOAZ: maybe the count forvar (and the other redundent forvars) forvar has a relative big overhead.. needs to be benchmarked
	for (int c = 0,				r = cols;	c < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Outer
	for (int c = 0,				r = 1;		c < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Inner
	for (int c = 0,				r = 0;		c < majorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//duu
	for (int c = 1,				r = 0;		r < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Inner
	for (int c = cols,			r = 0;		r < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r);	//Outer
	
	for (int c = majorDiagSize,			r = majorDiagSize + cols,	count = 0;	count < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Outer
	for (int c = majorDiagSize,			r = majorDiagSize + 1,		count = 0;	count < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Inner
	for (int c = majorDiagSize,			r = majorDiagSize,			count = 0;	count < majorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//dvv
	for (int c = majorDiagSize + 1,		r = majorDiagSize,			count = 0;	count < minorDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Inner
	for (int c = majorDiagSize + cols,	r = majorDiagSize,			count = 0;	count < outerDiagSize;	*(cooRowPtr++) = r, *(cooColPtr++) = c, ++c, ++r, ++count);	//Outer

	COOrdSparseMat* coo = new COOrdSparseMat(2 * majorDiagSize, 2 * majorDiagSize, nzAestimate, cooVal, cooRow, cooCol);
	A.build(*coo);
	delete coo;
}


void FlowOperator::construct(flowUV& UV, const cv::Mat& Du, const cv::Mat& Dv, const cv::Mat& Ix, const cv::Mat& Iy, const cv::Mat It, const OpticalFlowParams& params){
	////FMx/Fx const
	////FMx contains the FMxD mat on its main diagonal and FMxd on the -cols(in matlab its -rows) diagonal
	//FArray zeros(_cols, true, 0);
	//FArray ones(_cols, true, 1);
	//FArray nones(_cols, true, -1);

	///*
	//0  1  1  1       0  0  0  0
	//0  1  1  1  ->   1  1  1  1
	//0  1  1  1       1  1  1  1
	//0  1  1  1       1  1  1  1
	//*/
	//cv::Mat FMxD(_rows, _cols, OPTFLOW_TYPE);
	//memcpy(FMxD.ptr<float>(0), zeros.ptr, _cols * sizeof(float));
	//for(int i = 1; i < _rows; ++i)
	//	memcpy(FMxD.ptr<float>(i), ones.ptr, _cols * sizeof(float));
	///*
	//-1 -1 -1  0      -1 -1 -1 -1
	//-1 -1 -1  0  ->  -1 -1 -1 -1
	//-1 -1 -1  0      -1 -1 -1 -1
	//-1 -1 -1  0       0  0  0  0
	//*/
	////cv::Mat FMxd(_rows, _cols, OPTFLOW_TYPE, cv::Scalar(-1)); increases quality
	//cv::Mat FMxd(_rows, _cols, OPTFLOW_TYPE);
	//for(int i = 1; i < _rows; ++i)
	//	memcpy(FMxd.ptr<float>(i), nones.ptr, _cols * sizeof(float));
	//memcpy(FMxd.ptr<float>(0), zeros.ptr, _cols * sizeof(float));


	////FMy/Fy const

	///*
	// 0  0  0  0       0  1  1  1
	// 1  1  1  1  ->   0  1  1  1
	// 1  1  1  1       0  1  1  1
	// 1  1  1  1       0  1  1  1
	//*/
	//FArray line1(_cols, true, 1);
	//*(line1.ptr) = 0;	
	//cv::Mat FMyD(_rows, _cols, OPTFLOW_TYPE);
	//for(int i = 0; i < _rows; ++i)
	//	memcpy(FMyD.ptr<float>(i), line1.ptr, _cols * sizeof(float));
	//
	///*
	//-1 -1 -1 -1       -1 -1 -1  0
	//-1 -1 -1 -1   ->  -1 -1 -1  0
	//-1 -1 -1 -1       -1 -1 -1  0
	// 0  0  0  0       -1 -1 -1  0
	//*/
	//FArray line2(_cols, true, -1);
	//*(line2.ptr + _cols - 1) = 0;
	//cv::Mat FMyd(_rows, _cols, OPTFLOW_TYPE);
	//for(int i = 0; i < _rows; ++i)
	//	memcpy(FMyd.ptr<float>(i), line2.ptr, _cols * sizeof(float));



	cv::Mat tmp(_rows, _cols, OPTFLOW_TYPE);

	
	//cv::Mat flowU = UV.getU() + Du;
	cv::Mat flowU = cv::Mat((UV.getU() + Du).t()).reshape(1, FMxD.rows);
	UtilsMat::mulDiagVec<float>(FMxd, flowU, tmp, -_cols);
	cv::Mat txU = FMxD.mul(flowU) + tmp;
	
	
	//cv::Mat flowV = UV.getV() + Dv;
	cv::Mat flowV = cv::Mat((UV.getV() + Dv).t()).reshape(1, FMxD.rows);
	UtilsMat::mulDiagVec<float>(FMxd, flowV, tmp, -_cols);
	cv::Mat txV = FMxD.mul(flowV) + tmp;
	
	
	//UtilsMat::mulDiagVec<float>(FMyd, flowU, tmp, -_cols); increases quality
	UtilsMat::mulDiagVec<float>(FMyd, flowU, tmp, -1);
	cv::Mat tyU = FMyD.mul(flowU) + tmp;
	
	
	//UtilsMat::mulDiagVec<float>(FMyd, flowV, tmp, -_cols); increases quality
	UtilsMat::mulDiagVec<float>(FMyd, flowV, tmp, -1);
	cv::Mat tyV = FMyD.mul(flowV) + tmp;

	
	PenaltyFunctionCompute::compute(txU, params.getPfSpatialX(), params.getPenaltySigma(), params.getPfSpatialXDeriv());
	PenaltyFunctionCompute::compute(txV, params.getPfSpatialX(), params.getPenaltySigma(), params.getPfSpatialXDeriv());
	PenaltyFunctionCompute::compute(tyU, params.getPfSpatialY(), params.getPenaltySigma(), params.getPfSpatialYDeriv());
	PenaltyFunctionCompute::compute(tyV, params.getPfSpatialY(), params.getPenaltySigma(), params.getPfSpatialYDeriv());

	//FU, x part
	cv::Mat FUxD(_rows, _cols, OPTFLOW_TYPE);
	float* pxuDst1 = (float*)FUxD.data;
	float* pxud1 = (float*)FMxd.data;
	float* pxuX1 = ((float*)txU.data) + _cols;
	for (int i = _cols; i < _cells; ++i, ++pxud1)
		*(pxuDst1++) = *(pxuX1++) * *(pxud1) * *(pxud1);
	for (int i = 0; i < _cols; ++i)
		*(pxuDst1++) = 0;
	FUxD += FMxD.mul(txU.mul(FMxD));

	cv::Mat FUxu(_rows, _cols, OPTFLOW_TYPE);
	float* pxuDst2 = (float*)FUxu.data;
	float* pxud2 = (float*)FMxd.data;
	float* pxuD2 = ((float*)FMxD.data) + _cols;
	float* pxuX2 = ((float*)txU.data) + _cols;
	for (int i = _cols; i < _cells; ++i)
		*(pxuDst2++) = *(pxuX2++) * *(pxuD2++) * *(pxud2++);
	for (int i = 0; i < _cols; ++i)
		*(pxuDst2++) = 0;

	cv::Mat FUxl(_rows, _cols, OPTFLOW_TYPE);
	float* pxuDst3 = (float*)FUxl.data;
	float* pxud3 = (float*)FMxd.data;
	float* pxuD3 = ((float*)FMxD.data) + _cols;
	float* pxuX3 = ((float*)txU.data) + _cols;
	for (int i = _cols; i < _cells; ++i)
		*(pxuDst3++) = *(pxuX3++) * *(pxuD3++) * *(pxud3++);
	for (int i = 0; i < _cols; ++i)
		*(pxuDst3++) = 0;

	//FV, x part
	cv::Mat FVxD(_rows, _cols, OPTFLOW_TYPE);
	float* pxvDst1 = (float*)FVxD.data;
	float* pxvd1 = (float*)FMxd.data;
	float* pxvX1 = ((float*)txV.data) + _cols;
	for (int i = _cols; i < _cells; ++i, ++pxvd1)
		*(pxvDst1++) = *(pxvX1++) * *(pxvd1) * *(pxvd1);
	for (int i = 0; i < _cols; ++i)
		*(pxvDst1++) = 0;
	FVxD += FMxD.mul(txV.mul(FMxD));

	cv::Mat FVxu(_rows, _cols, OPTFLOW_TYPE);
	float* pxvDst2 = (float*)FVxu.data;
	float* pxvd2 = (float*)FMxd.data;
	float* pxvD2 = ((float*)FMxD.data) + _cols;
	float* pxvX2 = ((float*)txV.data) + _cols;
	for (int i = _cols; i < _cells; ++i)
		*(pxvDst2++) = *(pxvX2++) * *(pxvD2++) * *(pxvd2++);
	for (int i = 0; i < _cols; ++i)
		*(pxvDst2++) = 0;

	cv::Mat FVxl(_rows, _cols, OPTFLOW_TYPE);
	float* pxvDst3 = (float*)FVxl.data;
	float* pxvd3 = (float*)FMxd.data;
	float* pxvD3 = ((float*)FMxD.data) + _cols;
	float* pxvX3 = ((float*)txV.data) + _cols;
	for (int i = _cols; i < _cells; ++i)
		*(pxvDst3++) = *(pxvX3++) * *(pxvD3++) * *(pxvd3++);
	for (int i = 0; i < _cols; ++i)
		*(pxvDst3++) = 0;


	//FU, y part
	cv::Mat FUyD(_rows, _cols, OPTFLOW_TYPE);
	float* pyuDst1 = (float*)FUyD.data;
	float* pyud1 = (float*)FMyd.data;
	float* pyuX1 = ((float*)tyU.data) + 1;
	for (int i = 1; i < _cells; ++i, ++pyud1)
		*(pyuDst1++) = *(pyuX1++) * *(pyud1) * *(pyud1);
	*(pyuDst1++) = 0;
	FUyD += FMyD.mul(tyU.mul(FMyD));

	cv::Mat FUyu(_rows, _cols, OPTFLOW_TYPE);
	float* pyuDst2 = (float*)FUyu.data;
	float* pyud2 = (float*)FMyd.data;
	float* pyuD2 = ((float*)FMyD.data) + 1;
	float* pyuX2 = ((float*)tyU.data) + 1;
	for (int i = 1; i < _cells; ++i)
		*(pyuDst2++) = *(pyuX2++) * *(pyuD2++) * *(pyud2++);
	*(pyuDst2++) = 0;

	cv::Mat FUyl(_rows, _cols, OPTFLOW_TYPE);
	float* pyuDst3 = (float*)FUyl.data;
	float* pyud3 = (float*)FMyd.data;
	float* pyuD3 = ((float*)FMyD.data) + 1;
	float* pyuX3 = ((float*)tyU.data) + 1;
	for (int i = 1; i < _cells; ++i)
		*(pyuDst3++) = *(pyuX3++) * *(pyuD3++) * *(pyud3++);
	*(pyuDst3++) = 0;

	//FV, x part
	cv::Mat FVyD(_rows, _cols, OPTFLOW_TYPE);
	float* pyvDst1 = (float*)FVyD.data;
	float* pyvd1 = (float*)FMyd.data;
	float* pyvX1 = ((float*)tyV.data) + 1;
	for (int i = 1; i < _cells; ++i, ++pyvd1)
		*(pyvDst1++) = *(pyvX1++) * *(pyvd1) * *(pyvd1);
	*(pyvDst1++) = 0;
	FVyD += FMyD.mul(txU.mul(FMyD));

	cv::Mat FVyu(_rows, _cols, OPTFLOW_TYPE);
	float* pyvDst2 = (float*)FVyu.data;
	float* pyvd2 = (float*)FMyd.data;
	float* pyvD2 = ((float*)FMyD.data) + 1;
	float* pyvX2 = ((float*)tyV.data) + 1;
	for (int i = 1; i < _cells; ++i)
		*(pyvDst2++) = *(pyvX2++) * *(pyvD2++) * *(pyvd2++);
	*(pyvDst2++) = 0;

	cv::Mat FVyl(_rows, _cols, OPTFLOW_TYPE);
	float* pyvDst3 = (float*)FVyl.data;
	float* pyvd3 = (float*)FMyd.data;
	float* pyvD3 = ((float*)FMyD.data) + 1;
	float* pyvX3 = ((float*)tyV.data) + 1;
	for (int i = 1; i < _cells; ++i)
		*(pyvDst3++) = *(pyvX3++) * *(pyvD3++) * *(pyvd3++);
	*(pyvDst3++) = 0;

	cv::Mat FUD = FUxD + FUyD;
	cv::Mat FVD = FVxD + FVyD;

	FUD = -params.getAlpha() * FUD;
	FVD = -params.getAlpha() * FVD;
	FUyu = -params.getAlpha() * FUyu;
	FUyl = -params.getAlpha() * FUyl;
	FUxu = -params.getAlpha() * FUxu;
	FUxl = -params.getAlpha() * FUxl;
	FVyu = -params.getAlpha() * FVyu;
	FVyl = -params.getAlpha() * FVyl;
	FVxu = -params.getAlpha() * FVxu;
	FVxl = -params.getAlpha() * FVxl;

	cv::Mat pItx = (It.mul(Ix));
	cv::Mat pIty = (It.mul(Iy));

	//Perform linearization - note the change in It
	cv::Mat pIt(_rows, _cols, OPTFLOW_TYPE);
	It.copyTo(pIt);
	pIt += Ix.mul(Du) + Iy.mul(Dv);
	PenaltyFunctionCompute::compute(pIt, params.getPdData(), params.getPenaltySigma(), params.getPdDataDeriv());
	
	pItx = pItx.mul(pIt);
	pIty = pIty.mul(pIt);

	cv::Mat duu = pIt.mul(Ix.mul(Ix));
	cv::Mat dvv = pIt.mul(Iy.mul(Iy));
	cv::Mat duv = pIt.mul(Ix.mul(Iy));

	duu -= FUD;
	dvv -= FVD;
	
	this->bulidA(	(CRSSparseMat* const)&_A, 
			(const int)_cols, 
			(const int)_rows, 
			(const float* const)duu.data, 
			(const float* const)dvv.data, 
			(const float* const)duv.data, 
			(const float* const)(cv::Mat(-FUyu).data), 
			(const float* const)(cv::Mat(-FUyl).data), 
			(const float* const)(cv::Mat(-FUxu).data), 
			(const float* const)(cv::Mat(-FUxl).data), 
			(const float* const)(cv::Mat(-FVyu).data), 
			(const float* const)(cv::Mat(-FVyl).data), 
			(const float* const)(cv::Mat(-FVxu).data), 
			(const float* const)(cv::Mat(-FVxl).data));

	cv::Mat U = UV.getU();
	cv::Mat V = UV.getV();
	CRSSparseMat M;

	this->buildM(	M, 
					U.cols, 
					U.rows, 
					(const float* const)(FUD.data), 
					(const float* const)(FUyu).data, 
					(const float* const)(FUyl).data, 
					(const float* const)(FUxu).data, 
					(const float* const)(FUxl).data, 
					(const float* const)(FVD).data,
					(const float* const)(FVyu).data, 
					(const float* const)(FVyl).data, 
					(const float* const)(FVxu).data, 
					(const float* const)(FVxl).data);

	float* X = new float[_b.size()];
	memcpy(X,					 U.data, U.cols * U.rows * sizeof(float));
	memcpy(X + U.cols * U.rows , V.data, V.cols * V.rows * sizeof(float));

	M.MulVector(X,_b.ptr);
	
	delete[] X;

	for (int i = 0; i < pItx.cols * pItx.rows; ++i)
		_b.ptr[i] -= ((float*)pItx.data)[i];
	for (int i = 0; i < pIty.cols * pIty.rows; ++i)
		_b.ptr[pItx.cols * pItx.rows + i] -= ((float*)pIty.data)[i];

}

