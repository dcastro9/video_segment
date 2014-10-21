#pragma once
#include <fstream>
#include "cv.h"
#include "FArray.h"
#include "CRSSparseMat.h"
#include "limits.h"
#include <iostream>
using namespace std;
class UtilsDebug
{
public:
	static void printMat(const cv::Mat& mat, const string& filename){
		ofstream f;
		f.open (filename);
		for(int i = 0; i < mat.rows; ++i){
			for(int j = 0; j < mat.cols; ++j){
				f << i << ", " << j << ": " << *(((float*)mat.data) + (i * mat.cols) + j) << endl;
			}
		}
		f.close();
	}

	static void printMatInt(const cv::Mat& mat, const string& filename){
		ofstream f;
		f.open (filename);
		for(int i = 0; i < mat.rows; ++i){
			for(int j = 0; j < mat.cols; ++j){
				f << i << ", " << j << ": " << (int)*(((uchar*)mat.data) + (i * mat.cols) + j) << endl;
			}
		}
		f.close();
	}

	static void printCRSSparseMat(const CRSSparseMat& m, const string& filename){
		ofstream f;
		f.open (filename);
		for(int i = 0; i < m.dimR(); ++i){
			for(int r = m.rowPtr(i); r < m.rowPtr(i + 1); ++r){
				f << i << "\t" << m.colIdx(r) << "\t" << m.val(r) << endl;
			}
		}
		f.close();
	}

	static void printFArray(const FArray& arr, const string& filename){
		printArray(arr.ptr, arr.size(), filename);
	}

	static void printArray(const float* arr, const int size, const string& filename){
		ofstream f;
		f.open (filename);
		for(int i = 0; i < size; ++i){
			f << i << ": " << arr[i] << endl;
		}
		f.close();
	}

	static void minMaxMatFloat(cv::Mat& m){
		float min = (float)INT_MAX, max = (float)INT_MIN;
		for (int i = 0; i < m.rows * m.cols; ++i){
			if (((float*)m.data)[i] < min){
				min = ((float*)m.data)[i];
			}else if (((float*)m.data)[i] > max){
				max =  ((float*)m.data)[i];
			}
		}
	}


	static void printMatlab(const cv::Mat& mat, const string& filename){
		ofstream f;
		f.open (filename);
		for(int i = 0; i < mat.rows; ++i){
			for(int j = 0; j < mat.cols - 1; ++j){
				f <<  mat.ptr<float>(i)[j] << ", ";
			}
			f << mat.ptr<float>(i)[mat.cols - 1] <<  endl;
		}
		f.close();
	}
};

