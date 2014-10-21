#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <limits>
using namespace std;
#include "highgui.h" 
#include "cv.h"
#include <fstream>
#include "FlowUtils.h"

int ncols = 0;
#define MAXCOLS 60
int colorwheel[MAXCOLS][3];
void makecolorwheel();

float UtilsFlow::FindMaxLength(cv::Mat& U,cv::Mat& V)
{
	float maxLength=-numeric_limits<float>::max();
	for (int y=0; y < U.rows;y++)
	{
		for (int x=0; x < U.cols;x++)
		{
			float u,v;
			u = U.at<float>(y,x);
			v = V.at<float>(y,x);
			if (fabs(u)>1e9 || fabs(v)>1e9)
				continue;
			float l=sqrt(u*u+v*v);
			if (l>maxLength)
				maxLength=l;
		}
	}
	return maxLength;
}

void UtilsFlow::GetFlowImage(cv::Mat& U,cv::Mat& V, cv::Mat& dst, float maxMotion)
{
	const float pi=3.1415926535897932384626433832795f;
	dst.create(U.rows,U.cols, CV_8UC3);
	dst.setTo(cv::Scalar(0));
	float maxLength=FindMaxLength(U,V);

	if (maxMotion>0.0f)
		maxLength=maxMotion;

	if (maxLength==0.0f)
		maxLength=1.0f;

	if (ncols==0)
		makecolorwheel();

	for (int y = 0; y < U.rows; y++)
	{
		for (int x = 0; x < U.cols; x++)
		{
			cv::Vec3b out = dst.at<cv::Vec3b>(y,x);
			float u,v;
			u = U.at<float>(y,x);
			v = V.at<float>(y,x);			
			if (fabs(u)<1e9 && fabs(v)<1e9)
			{
				u=u/maxLength;
				v=v/maxLength;
				float rad = sqrt(u * u + v * v);
				float a = atan2(-v, -u) / pi;
				float fk = (a + 1.0f) / 2.0f * (ncols-1);
				int k0 = (int)fk;
				int k1 = (k0 + 1) % ncols;
				float f = fk - k0;
				//f = 0; // uncomment to see original color wheel
				for (int b = 0; b < 3; b++) {
					float col0 = colorwheel[k0][b] / 255.0f;
					float col1 = colorwheel[k1][b] / 255.0f;
					float col = (1 - f) * col0 + f * col1;
					if (rad <= 1)
						col = 1 - rad * (1 - col); // increase saturation with radius
					else
					{
						col *= .75; // out of range
						out.val[b]=(int)(255.0 * col);
					}
				}				
			}
			dst.at<cv::Vec3b>(y,x) = out;
		}
	}
}


void UtilsFlow::DrawFlow(cv::Mat& U,cv::Mat& V, string windowname)
{
	cv::Mat flow;
	GetFlowImage(U,V, flow);
	cv::cvtColor(flow, flow, CV_BGR2RGB);
	//cv::imshow(windowname, flow);
	ShowManyImages(windowname,1,flow);
	cvWaitKey(1);			
	flow.release();
}


void UtilsFlow::DrawFlow2(cv::Mat& du,cv::Mat& u,cv::Mat& dv,cv::Mat& v, string windowname)
{
	cv::Mat tempsumU(u.cols,u.rows, u.type());
	cv::Mat tempsumV(u.cols,u.rows, u.type());

	cv::add(u,du,tempsumU);
	cv::add(v,dv,tempsumV);

	DrawFlow(tempsumU,tempsumV, windowname);

	tempsumU.release();
	tempsumV.release();

}

void UtilsFlow::ShowManyImages(string title, int nArgs, ...) {

	// img - Used for getting the arguments 
	cv::Mat img;
	// DispImage - the image in which input images are to be copied
	cv::Mat DispImage;
	int size;
	int i;
	int m, n;
	int x, y;
	// w - Maximum number of images in a row 
	// h - Maximum number of images in a column 
	int w, h;
	// scale - How much we have to resize the image
	float scale;
	int max;
	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying 
	if(nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if(nArgs > 12) {
		printf("Number of arguments too large....\n");
		return;
	}
	// Determine the size of the image, 
	// and the number of rows/cols 
	// from number of arguments 
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}
	va_list args1;
	va_start(args1, nArgs);
	// Create a new 3 channel image
	img = va_arg(args1, cv::Mat);
	if(img.channels() == 1)
		DispImage.create(60 + size*h, 100 + size*w, CV_MAKETYPE(img.depth(),1));
	else
		DispImage.create(60 + size*h, 100 + size*w, CV_MAKETYPE(img.depth(),3));
	va_list args;
	va_start(args, nArgs);
	// Used to get the arguments passed
	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		img = va_arg(args, cv::Mat);
		// Find the width and height of the image
		x = img.cols;
		y = img.rows;
		// Find whether height or width is greater in order to resize the image
		max = (x > y)? x: y;
		// Find the scaling factor to resize the image
		scale = (float) ( (float) max / size );
		// Used to Align the images
		if( i % w == 0 && m!= 20) {
			m = 20;
			n+= 20 + size;
		}
		// Set the image ROI to display the current image
		cv::Mat ROIDisp = DispImage(cv::Rect(m, n, (int)( x/scale ), (int)( y/scale )));
		// Resize the input image and copy the it to the Single Big Image
		cv::resize(img, ROIDisp, cv::Size((int)( x/scale ), (int)( y/scale )));
	}
	// Create a new window, and show the Single Big Image
	cv::namedWindow( title, 1 );
	cv::imshow( title, DispImage);
	cv::waitKey(1);
	//cvDestroyWindow(title);
	// End the number of arguments
	va_end(args);
	// Release the Image Memory
	DispImage.release();
}

void UtilsFlow::setcols(int r, int g, int b, int k)
{
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}

void UtilsFlow::makecolorwheel()
{
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	if (ncols > MAXCOLS)
		exit(1);
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255,			255*i/RY,		0,				k++);
	for (i = 0; i < YG; i++) setcols(255-255*i/YG,	255,			0,				k++);
	for (i = 0; i < GC; i++) setcols(0,				255,			255*i/GC,		k++);
	for (i = 0; i < CB; i++) setcols(0,				255-255*i/CB,	255,			k++);
	for (i = 0; i < BM; i++) setcols(255*i/BM,		0,				255,			k++);
	for (i = 0; i < MR; i++) setcols(255,			0,				255-255*i/MR,	k++);
}


bool UtilsFlow::ReadFlowFile(const string &filename,cv::Mat &U,cv::Mat &V)
{
	std::ifstream file;
	file.open(filename.c_str(),ios::binary);
	if (!file)
		return false;

	/*streambuf * pbuf;
	pbuf = file.rdbuf();*/

	static const string bla = "PIEH";
	char tag[5];
	tag[4] = NULL;
	//pbuf->sgetn(tag, 4);
	file.read(tag, 4);

	if (bla.compare(tag) != 0)
	{
		file.close();
		return false;
	}

	//pbuf->sgetn(tag, 4);
	file.read(tag, 4);
	int cols = *(int*)tag;
	//pbuf->sgetn(tag, 4);
	file.read(tag, 4);
	int rows = *(int*)tag;
	
	U = cv::Mat(rows, cols, CV_32FC1);
	V = cv::Mat(rows, cols, CV_32FC1);
	char* buf = new char[/*2 * */sizeof(float)];
	//char* buf2 = buf + sizeof(float);
	//float* v = (float*)V.data;
	//float* u = (float*)U.data;
	//for (int i = 0; i < cols * rows; ++i, ++u, ++v){
	//	pbuf->sgetn (buf, 2 * sizeof(float));
	//	//file.read(buf, sizeof(float));
	//	*u = *(float*)buf;
	//	//file.read(buf, sizeof(float));
	//	*v = *(float*)buf2;
	//}

	for (int y = 0; y < U.rows; ++y)
	{
		for (int x = 0; x < U.cols; ++x)
		{
			file.read(buf, sizeof(float));
			U.at<float>(y,x) = *(float*)buf;
			file.read(buf, sizeof(float));
			V.at<float>(y,x) = *(float*)buf;
		}
	}
	file.close();

	delete[] buf;

	return true;
}

bool UtilsFlow::WriteFlowFile(const string filename,cv::Mat &U,cv::Mat &V)
{
	std::ofstream file;
	file.open(filename.c_str(), fstream::binary);

	if (!file)
	{
		//ERROR
		return false;
	}
	file.write("PIEH", 4);
	int cols=U.cols;
	int rows=U.rows;
	file.write((char*)&cols, 4);
	file.write((char*)&rows, 4);

	for (int y=0;y<U.rows;y++)
	{
		for (int x=0;x<U.cols;x++)
		{
			float u=U.at<float>(y,x);
			float v=V.at<float>(y,x);
			file.write((char*)&u, 4);
			file.write((char*)&v, 4);
		}
	}
	file.close();
	return true;
}

