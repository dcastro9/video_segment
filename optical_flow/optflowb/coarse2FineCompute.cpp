#include "coarse2FineCompute.h"
#include "Decomposition.h"
#include "FlowOperator.h"
#include "FlowUtils.h"
#include "CRSSparseMat.h"
#include "WeightedMedianFilter.h"
#include "Defs.h"
#include "UtilsMat.h"

#include <ctime>
#include <iostream>


void OpticalFlow::WarpImage(const cv::Mat& im, const cv::Mat& u,const cv::Mat& v, cv::Mat& dst){
	
	//remap requires 2 maps with absolute coordinates to move the pixels - p(1,1) -> m(1,1) stays in place p(1,1), 
	//but p(1,1) -> m(2,2) = p(2,2). thus we're fixing the uv flow mats to represent absolute movement of pixels
	//and the remapping with cubuc interpolation
	cv::Mat x(u.rows, u.cols, u.type());
	cv::Mat y(v.rows, v.cols, v.type());

	float* pU = (float*)u.data;
	float* pV = (float*)v.data;
	float* pX = (float*)x.data;
	float* pY = (float*)y.data;
	for( int i = 0; i < im.rows; ++i )
		for( int j = 0; j < im.cols; ++j){
			*(pX++) = *(pU++) + j;
			*(pY++) = *(pV++) + i;
		}
	cv::remap(im, dst, x, y, cv::INTER_CUBIC);
}


flowUV* OpticalFlow::calculate(const cv::Mat& Im1, const cv::Mat& Im2, const OpticalFlowParams& params, flowUV* oldFlow)
{
	cv::Mat Im1p, Im2p;
	if(params.isUseTextureDecomposition()){
#if OPTFLOW_VERBOSE
		cout << "Coarse2FineFlow: Texture Decomposition..." ;
		std::clock_t decstart = std::clock();
#endif
		Decomposition::structureTextureDecompositionRof(Im1, Im2, Im1p, Im2p, NULL, NULL, params.getTDtheta(), params.getTDnIters(), params.getTDalp(), params.getDisplayTextureDecompositionOutput());
#if OPTFLOW_VERBOSE
	cout<<" --- "<< (std::clock() - decstart) / (double)CLOCKS_PER_SEC <<'\n';
#endif
	}else{
		Im1p = Im1;
		Im2p = Im2;
	}
	GaussPyramid Pyramid1(Im1p, params.getPyramidLevels(), params.getPyramidSpacing());	
	GaussPyramid Pyramid2(Im2p, params.getPyramidLevels(), params.getPyramidSpacing());

	//flow result
	flowUV* UV;
	if (oldFlow == NULL){
		UV = new flowUV(Pyramid2[0].rows, Pyramid2[0].cols);
	}else{
		UV = oldFlow;
		(*UV).reshape(Pyramid2[0].rows, Pyramid2[0].cols);
	}

#if OPTFLOW_VERBOSE
	cout << "OpticalFlow: calculating flow ..." << endl;
#endif

	// now iterate from the top level to the bottom
	for(int k = 0; k < Pyramid1.getNlevels(); ++k)
	{	
		cv::Mat curr1Level = Pyramid1[k];
		cv::Mat curr2Level = Pyramid2[k];
#if OPTFLOW_VERBOSE
		cout << "Pyramid level " << k << " |- " << "cols: " << curr1Level.cols << " , rows: " << curr1Level.rows << endl;
#endif
		if (curr1Level.channels() >= 3){
			cv::cvtColor(curr1Level, curr1Level, CV_BGR2GRAY);
			cv::cvtColor(curr2Level, curr2Level, CV_BGR2GRAY);
		}
		if (k != 0){
			UV->reshape(curr1Level.rows, curr1Level.cols);
		}	
#if OPTFLOW_VERBOSE					
		std::clock_t start = std::clock();
#endif
		
		baseCalculate(curr1Level, curr2Level, *UV, params);
#if OPTFLOW_VERBOSE	
		std::cout<<"baseCalculate took: "<< ( std::clock() - start ) / (double)CLOCKS_PER_SEC <<" secs"<<endl;
#endif

	}
	return UV;
}


void getDXsCV(const cv::Mat src1, cv::Mat dest_dx, cv::Mat dest_dy){		
	static double x[5] = {0.0833, -0.6667, 0, 0.6667, -0.0833};					
	static double y[5] = {0.0833, -0.6667, 0, 0.6667, -0.0833};
 
	//x derivative
	cv::Mat weickertX(1, 5, CV_64FC1, x ); // 64FC1 for double
	cv::filter2D(src1, dest_dx, dest_dx.depth(), weickertX, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
	
	//y derivative
	cv::Mat weickertY(5, 1, CV_64FC1, y ); // 64FC1 for double
	cv::filter2D(src1, dest_dy, dest_dy.depth(), weickertY, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
}


void OpticalFlow::baseCalculate(cv::Mat& Im1, cv::Mat& Im2, flowUV& UV, const OpticalFlowParams& params){
	int rows = Im1.rows;
	int cols = Im1.cols;

	FlowOperator flowOp(rows, cols);
	FArray X0(2 * rows * cols, false);	

	FArray dUdV(2 * rows * cols, true, 0);
	cv::Mat Ix1(rows, cols, OPTFLOW_TYPE);
	cv::Mat Iy1(rows, cols, OPTFLOW_TYPE); 
	cv::Mat Ix(rows, cols, OPTFLOW_TYPE);
	cv::Mat Iy(rows, cols, OPTFLOW_TYPE); 
	getDXsCV(Im1, Ix1, Iy1);
	for (int i = 0; i < params.getIters(); ++i){
		cv::Mat Ix2(rows, cols, OPTFLOW_TYPE);
		cv::Mat Iy2(rows, cols, OPTFLOW_TYPE); 
		cv::Mat It(rows, cols, OPTFLOW_TYPE); 

		cv::Mat im2Warpped(rows, cols, Im1.type());
		WarpImage(Im2, UV.getU(), UV.getV(), im2Warpped);	
		
		getDXsCV(im2Warpped, Ix2, Iy2);
		Ix = params.getWeightedDeriveFactor() * (Ix1 + Ix2);
		Iy = params.getWeightedDeriveFactor() * (Iy1 + Iy2);
		cv::subtract(im2Warpped, Im1, It);

		if (params.getDisplayDerivativs()){
			cv::imshow("Derivative Ix", Ix);
			cv::imshow("Derivative Iy", Iy);
			cv::waitKey(1);
		}
		
		cv::Mat Du(rows, cols, OPTFLOW_TYPE, cv::Scalar(0));
		cv::Mat Dv(rows, cols, OPTFLOW_TYPE, cv::Scalar(0));


		for (int j = 0; j < params.getLinearIters(); ++j){
#if OPTFLOW_VERBOSE
			cout << "solving Ax=b with SOR ";
			clock_t start = std::clock();	
#endif		
			flowOp.construct(UV, Du, Dv, Ix, Iy, It, params);
			
			memcpy(X0.ptr, UV.getU().data, rows * cols * sizeof(float));
			memcpy(X0.ptr + (rows * cols), UV.getV().data, rows * cols * sizeof(float));
			//UtilsDebug::printCRSSparseMat(flowOp.getA(), "aaaa.txt");
			if (params.getCheckResidualTolerance()){
				LinearSolver::sparseMatSor(flowOp.getA(), X0 ,dUdV, flowOp.getb(), params.getOverRelaxation(), params.getSorIters(), params.getResidualTolerance());
			}else{
				//LinearSolver::multigrid(10,10,flowOp.getA(),flowOp.getb(), params.getResidualTolerance(), dUdV, 20, 20, LinearSolver::vCycle);
				LinearSolver::sparseMatSorNoResidual(flowOp.getA(), X0 ,dUdV, flowOp.getb(), params.getOverRelaxation(), params.getSorIters());
			}
#if OPTFLOW_VERBOSE
		std::cout<<" --- "<< (std::clock() - start) / (double)CLOCKS_PER_SEC <<'\n';
#endif

#if OPTFLOW_DEBUG
			for(int i = 0; i < dUdV.size(); ++i){
				if (!(dUdV.ptr[i] == dUdV.ptr[i])){
					cout << "ERROR - NAN";
				}
			}
#endif

			UtilsMat::clamp(dUdV, -1, 1);

			memcpy(Du.data, dUdV.ptr, rows * cols * sizeof(float));
			memcpy(Dv.data, dUdV.ptr + (rows * cols), rows * cols * sizeof(float));
			
			flowUV UV0(UV);
			UV.getU() += Du;
			UV.getV() += Dv;

			cv::Mat tmpU, tmpV;
			UV.getU().copyTo(tmpU);
			UV.getV().copyTo(tmpV);

			WeightedMedianFilter::computeMedianFilter(UV.getU(), UV.getV(), Im1, Im2, params.getMedianFilterRadius());

			Du = UV.getU() - UV0.getU();
			Dv = UV.getV() - UV0.getV();

			UV0.getU().copyTo(UV.getU());
			UV0.getV().copyTo(UV.getV());

			UV0.getU() += Du;
			UV0.getV() += Dv;
			if (params.isDisplay())
				UtilsFlow::DrawFlow(UV0.getU(), UV0.getV(), "Flow");
		}
		UV.getU() += Du;
		UV.getV() += Dv;
	}
}

