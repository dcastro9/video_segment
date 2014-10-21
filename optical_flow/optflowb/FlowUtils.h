#pragma once
using namespace std;
class UtilsFlow
{
public:

	static bool ReadFlowFile(const string &filename,cv::Mat &U,cv::Mat &V);
	static bool WriteFlowFile(const string filename,cv::Mat &U,cv::Mat &V);

	static void GetFlowImage(cv::Mat& U,cv::Mat& V,cv::Mat & dst, float maxMotion=0.001f); 
	static void ShowManyImages(string title, int nArgs, ...);
	static void DrawFlow(cv::Mat& U,cv::Mat& V, string windowname = "flow");
	static void DrawFlow2(cv::Mat& du,cv::Mat& u,cv::Mat& dv,cv::Mat& v, string windowname = "flow");

private:
	static void makecolorwheel();
	static void setcols(int r, int g, int b, int k);
	static float FindMaxLength(cv::Mat& U,cv::Mat& V);
};