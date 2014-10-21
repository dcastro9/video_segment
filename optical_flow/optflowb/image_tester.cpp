#include "coarse2FineCompute.h"
#include <ctime>
#include "FlowError.h"
#include "OpticalFlowParams.h"
#include "FlowUtils.h"
#include "UtilsDebug.h"
#include "Defs.h"
#include <iostream>
using namespace std;

void calculateFlowPerformence(string files[]){
	cv::Mat image1 = cv::imread(files[1]);
	cv::Mat image2 = cv::imread(files[2]);
	
	cv::Mat image1Gray(image1.rows, image1.cols, CV_MAKETYPE(image1.depth(), 1));
	cv::Mat image2Gray(image2.rows, image2.cols, CV_MAKETYPE(image2.depth(), 1));
	cv::cvtColor(image1, image1Gray, CV_BGR2GRAY);
	cv::cvtColor(image2, image2Gray, CV_BGR2GRAY);
	
	image1.release();
	image2.release();

	cv::Mat image1GrayFloat(image1.rows, image1Gray.cols, OPTFLOW_TYPE);
	cv::Mat image2GrayFloat(image2.rows, image2Gray.cols, OPTFLOW_TYPE);
	image1Gray.convertTo(image1GrayFloat, OPTFLOW_TYPE);
	image2Gray.convertTo(image2GrayFloat, OPTFLOW_TYPE);

	image1Gray.release();
	image2Gray.release();

	OpticalFlowParams GNC1params(3, 5, 0.5f, 0.5f, false, 5, 100, 1.9f, false, 0.01f, 3, 1, 0.001f, 
									PenaltyFunctionCompute::Quadratic, 
									PenaltyFunctionCompute::Second, 
									PenaltyFunctionCompute::Quadratic,
									PenaltyFunctionCompute::Second,
									PenaltyFunctionCompute::Quadratic, 
									PenaltyFunctionCompute::Second,
									true, 1.0f/8.0f, 100, 0.95f, false, false);

	flowUV* UV = OpticalFlow::calculate(image1GrayFloat, image2GrayFloat, GNC1params);

	delete UV;
}

float* calculateFlow(string files[], bool verbose = false, bool ignoreErr = false)
{ 
	cv::Mat tU;
	cv::Mat tV;
	if (!ignoreErr){
		UtilsFlow::ReadFlowFile(files[0], tU, tV);
		if (verbose)
			UtilsFlow::DrawFlow(tU, tV, "Ground Truth");
	}
	flowUV GT(tU, tV);
	

	cv::Mat image1 = cv::imread(files[1]);
	cv::Mat image2 = cv::imread(files[2]);
	
	cv::Mat image1Gray(image1.rows, image1.cols, CV_MAKETYPE(image1.depth(), 1));
	cv::Mat image2Gray(image2.rows, image2.cols, CV_MAKETYPE(image2.depth(), 1));
	cv::cvtColor(image1, image1Gray, CV_BGR2GRAY);
	cv::cvtColor(image2, image2Gray, CV_BGR2GRAY);
	
	image1.release();
	image2.release();

	cv::Mat image1GrayFloat(image1.rows, image1Gray.cols, OPTFLOW_TYPE);
	cv::Mat image2GrayFloat(image2.rows, image2Gray.cols, OPTFLOW_TYPE);
	image1Gray.convertTo(image1GrayFloat, OPTFLOW_TYPE);
	image2Gray.convertTo(image2GrayFloat, OPTFLOW_TYPE);

	image1Gray.release();
	image2Gray.release();

	OpticalFlowParams GNC1params(3, 5, 0.5f, 0.5f, false, 5, 100, 1.9f, false, 0.01f, 3, 1, 0.001f, 
									PenaltyFunctionCompute::Quadratic, 
									PenaltyFunctionCompute::Second, 
									PenaltyFunctionCompute::Quadratic,
									PenaltyFunctionCompute::Second,
									PenaltyFunctionCompute::Quadratic, 
									PenaltyFunctionCompute::Second,
									true, 1.0f/8.0f, 100, 0.95f, false, false);

	//OpticalFlowParams GNC2params(3, 2, 0.8f, 0.5f, false, 5, 100, 1.9f, false, 0.01f, 3, 1, 0.001f , 
	//								PenaltyFunctionCompute::GeneralizedCharbonnier, 
	//								PenaltyFunctionCompute::Second, 
	//								PenaltyFunctionCompute::GeneralizedCharbonnier,
	//								PenaltyFunctionCompute::Second,
	//								PenaltyFunctionCompute::GeneralizedCharbonnier,
	//								PenaltyFunctionCompute::Second,
	//								false, 1.0f/8.0f, 100, 0.95f, false, true);
	size_t start;
	if(verbose){
		cout << "Image tester: calculating flow..." << endl;
		 start = std::clock();
	}
	//calculate GNC-1 flow
	flowUV* UV = OpticalFlow::calculate(image1GrayFloat, image2GrayFloat, GNC1params);
	//cout << "Image tester: calculating GNC flow... " << endl;
	////calculate GNC-2 flow
	//UV = OpticalFlow::calculate(image1GrayFloat, image2GrayFloat, GNC2params, UV);
	if(verbose)
		cout << "Image tester: ENDED calculating flow in " << (std::clock() - start) << endl;

	image1GrayFloat.release();
	image2GrayFloat.release();

	if(verbose){
		cout << "Image tester: drawing flow..." << endl;
		UtilsFlow::DrawFlow(UV->getU(), UV->getV(), "Flow");
		cout << "Image tester: calculating error..." << endl;
	}
	float* err = ignoreErr? 0 : FlowError::calcError(*UV, GT, verbose);
	if(verbose)
		cout << "Image tester: AAE " << err[0] << " STD " << err[1] << " average end point error " << err[2] << endl;

	//UtilsFlow::WriteFlowFile("flowTest.flo", UV->getU(), UV->getV());

	delete UV;
	if(verbose){
		cout << "End." << endl;
		cv::waitKey(0);
	}

	return err;
} 

int main (int argc,char** argv) 
{ 

	cv::Mat kernelX = (cv::Mat_<float>(1,3) << -1, 0, 1);
	for (int j=0; j<1;j++)
	{
		float* data=kernelX.ptr<float>(j);
		for (int i=0; i<3;i++)
		{ 
			cout<< (float)data[i] << " "; 
		}
		cout<< endl;
	}


	string files[8][3];
	string name[8] = {"Dimetrodon", "Grove2", "Grove3", "Hydrangea", "RubberWhale", "Urban2", "Urban3", "Venus"};


	files[0][0] = "Data\\Dimetrodon\\flow10.flo";
	files[0][1] = "Data\\Dimetrodon\\frame10.png";
	files[0][2] = "Data\\Dimetrodon\\frame11.png";
 
	files[1][0] = "Data\\Grove2\\flow10.flo";
	files[1][1] = "Data\\Grove2\\frame10.png";
	files[1][2] = "Data\\Grove2\\frame11.png";
 
	files[2][0] = "Data\\Grove3\\flow10.flo";
	files[2][1] = "Data\\Grove3\\frame10.png";
	files[2][2] = "Data\\Grove3\\frame11.png";
 
	files[3][0] = "Data\\Hydrangea\\flow10.flo";
	files[3][1] = "Data\\Hydrangea\\frame10.png";
	files[3][2] = "Data\\Hydrangea\\frame11.png";
 
	files[4][0] = "Data\\RubberWhale\\flow10.flo";
	files[4][1] = "Data\\RubberWhale\\frame10.png";
	files[4][2] = "Data\\RubberWhale\\frame11.png";
 
	files[5][0] = "Data\\Urban2\\flow10.flo";
	files[5][1] = "Data\\Urban2\\frame10.png";
	files[5][2] = "Data\\Urban2\\frame11.png";
 
	files[6][0] = "Data\\Urban3\\flow10.flo";
	files[6][1] = "Data\\Urban3\\frame10.png";
	files[6][2] = "Data\\Urban3\\frame11.png";
 
	files[7][0] = "Data\\Venus\\flow10.flo";
	files[7][1] = "Data\\Venus\\frame10.png";
	files[7][2] = "Data\\Venus\\frame11.png";
	if (true){
		size_t start = std::clock();
		calculateFlowPerformence(files[4]);
		cout << "Took " << (std::clock() - start) / 1000.0f << endl;
		//calculateFlow(files[4], true, false);
	}else{
 
		std::fstream file;
		file.open("Results.txt", fstream::in | fstream::out | fstream::trunc);
		for (int i = 0; i < 8; i++)
		{
			size_t start = std::clock();

			float* err = calculateFlow(files[i]);
			file << name[i] << ": Took " << (std::clock() - start) / 1000.0f<< "  AAE " << err[0] << "  STD " << err[1] << "  EPE " << err[2] << endl;
			delete[] err;
			cout << i + 1 << "/" << 8 << " Done" << endl;
		}
		file.close();
	}
}
