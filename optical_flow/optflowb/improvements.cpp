//#include <vector>
//#include <algorithm>
//#include "highgui.h" 
//#include "cv.h"
//#include "IplImageIterator.h"
//#include <cmath>
//
//
//#define ELEM_SWAP(a,b) { double t=(a);(a)=(b);(b)=t; }
//double quick_select(double arr[], int n)
//{
//	int low, high ;
//	int median;
//	int middle, ll, hh;
//	low = 0 ; high = n-1 ; median = (low + high) / 2;
//	for (;;) {
//		if (high <= low) /* One element only */
//			return arr[median] ;
//		if (high == low + 1) { /* Two elements only */
//			if (arr[low] > arr[high])
//				ELEM_SWAP(arr[low], arr[high]) ;
//			return arr[median] ;
//		}
//		/* Find median of low, middle and high items; swap into position low */
//		middle = (low + high) / 2;
//		if (arr[middle] > arr[high]) ELEM_SWAP(arr[middle], arr[high]) ;
//		if (arr[low] > arr[high]) ELEM_SWAP(arr[low], arr[high]) ;
//		if (arr[middle] > arr[low]) ELEM_SWAP(arr[middle], arr[low]) ;
//		/* Swap low item (now in position middle) into position (low+1) */
//		ELEM_SWAP(arr[middle], arr[low+1]) ;
//		/* Nibble from each end towards middle, swapping items when stuck */
//		ll = low + 1;
//		hh = high;
//		for (;;) {
//			do ll++; while (arr[low] > arr[ll]) ;
//			do hh--; while (arr[hh] > arr[low]) ;
//			if (hh < ll)
//				break;
//			ELEM_SWAP(arr[ll], arr[hh]) ;
//		}
//		/* Swap middle item (in position low) back into correct position */
//		ELEM_SWAP(arr[low], arr[hh]) ;
//		/* Re-set active partition */
//		if (hh <= median)
//			low = ll;		
//			if (hh >= median)
//				high = hh - 1;
//	}
//}
//#undef ELEM_SWAP
//
//
//cv::Mat medianFilter(cv::Mat src, int n)
//{
//	CvScalar s;
//	std::vector<double> neighbours(n*n);
//	cv::Mat ret= src.clone();
//	int edge=n/2;	
//	for (int y=edge;y<ret.rows-edge;y++)
//	{
//		for (int x=edge;x<ret.cols-edge;x++)
//		{
//			neighbours.clear();
//			for (int j=-edge;j<=edge;j++)
//			{
//				for (int i=-edge;i<=edge;i++)
//				{
//					src.at<cv::Scalar>(y+j,x+i);
//					//s=cvGet2D(src,y+j,x+i);
//					neighbours.push_back(s.val[0]);
//				}
//			}
//			s.val[0]=quick_select(&neighbours[0],n*n);
//			ret.at<cv::Scalar>(y,x) = s;
//			//cvSet2D(ret,y,x,s);
//		}
//	}
//	return ret;	
//}
//
//
//void ShowImage(char *title,IplImage *img)
//{
//	cvNamedWindow( title, 1 );
//	cvShowImage( title, img);
//
//	cvWaitKey();
//	cvDestroyWindow(title);
//}
//void ShowImage2(char *title,IplImage *img)
//{
//	cvNamedWindow( title, 1 );
//	cvShowImage( title, img);    
//}
