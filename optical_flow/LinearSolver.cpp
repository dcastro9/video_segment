#include "LinearSolver.h"
#include <string.h>
#include <math.h>
#include <iostream>
#define DEFECT_FACTOR 10
#define DEBUG_SOLVER false

void LinearSolver::sparseMatSor(CRSSparseMat& A, FArray& X0, FArray& X, FArray& B, const float w, const int numOfIterations, const float residualTolerance){	
#if OPTFLOW_STRICT
	if(B.size != X.size){
		printf("Error solve:B and X should be vectors of the same size \n");
		exit(-1); //BOAZ: KAKA!
	}
	if(A.rows() != A.cols() || A.rows() != X.size){
		printf("Error solve:A should be nxn matrix and X should be vector of size n \n");
		exit(-1); //BOAZ: KAKA!
	}
#endif
	if (!(X0.ptr == X.ptr))
		memcpy(X.ptr, X0.ptr, X0.size() * sizeof(float));
	int rows = A.dimR();
	for(int k = 0; k < numOfIterations; ++k){
		if ((k % 5) == 0)
			if (residual(A, X, B) < residualTolerance)	return;

		int* pRowPtr = A.rowPtr();
		int* pColIdx = A.colIdx();
		float* pVal = A.val();
		for (int i = 0; i < rows; ++i, ++pRowPtr)
		{
			float temp = 0;
			float diag = 0;
			for (int p = *pRowPtr; p < *(pRowPtr + 1); ++p)
			{
				int j = *(pColIdx++);
				if (j == i) { diag = *(pVal++); continue;}
				temp += *(pVal++) * X.ptr[j];
			}

			temp = (B.ptr[i] - temp) / diag;
			X.ptr[i] += w * (temp - X.ptr[i]);
		}	
	}
}

void LinearSolver::sparseMatSorNoResidual(CRSSparseMat& A, FArray& X0, FArray& X, FArray& B, const float w, const int numOfIterations){	
#if OPTFLOW_STRICT
	if(B.size != X.size){
		printf("Error solve:B and X should be vectors of the same size \n");
		exit(-1); //BOAZ: KAKA!
	}
	if(A.rows() != A.cols() || A.rows() != X.size){
		printf("Error solve:A should be nxn matrix and X should be vector of size n \n");
		exit(-1); //BOAZ: KAKA!
	}
#endif
	if (!(X0.ptr == X.ptr))
		memcpy(X.ptr, X0.ptr, X0.size() * sizeof(float));
	int rows = A.dimR();
	float nW = 1 - w;
	for(int k = 0; k < numOfIterations; ++k){
		int* pRowPtr = A.rowPtr();
		int* pColIdx = A.colIdx();
		float* pVal = A.val();
		float* pX = X.ptr;
		for (int i = 0; i < rows; ++i, ++pRowPtr)
		{
			float temp = 0;
			float diag = 0;
			for (int p = *pRowPtr; p < *(pRowPtr + 1); ++p)
			{
				int j = *(pColIdx++);
				if (j == i) { diag = *(pVal++); continue;}
				temp += *(pVal++) * X.ptr[j];
			}
			temp = (B.ptr[i] - temp) / diag;
			X.ptr[i] += w * (temp - X.ptr[i]);
			//*(pX++) = nW * (*pX) + (w * temp);
		}	
	}
}

double LinearSolver::residual(CRSSparseMat& A, FArray& X, FArray& B){
	int total = 0;
	double res = 0;
	int rows = A.dimR();

	int* pRowPtr = A.rowPtr();
	int* pColIdx = A.colIdx();
	float* pVal = A.val();

	for (int i = 0; i < rows; ++i, ++pRowPtr)
	{
		float temp = -B.ptr[i];
		for (int p = *pRowPtr; p < *(pRowPtr + 1); ++p)
		{
			temp += *(pVal++) * X.ptr[*(pColIdx++)];
		}
		res += temp * temp;
	}
	return sqrt(res);
}

//void LinearSolver::multigrid(const int maxIterations, const int maxLevels, CRSSparseMat& A, FArray& B, const float w, FArray& X, const int preIterations, const int postIterations, const CycleType cycleType){		
//	for(int i = 0; i < maxIterations; ++i){
//		multigridIteration(maxLevels, A, B, w, X, preIterations, postIterations, cycleType);
//	}	
//}
//
//void LinearSolver::multigridIteration(const int currentLevel, CRSSparseMat& A, FArray& B, const float w, FArray& X, const int preIterations, const int postIterations, const CycleType cycleType){
//	int h = X.size();
//#if OPTFLOW_STRICT
//	if(	(h < 2 ) || ((h % 2) != 0) ){
//		cout << "Error multigrid dimension  must be even number >= 2 " << endl;
//		return ;
//	}
//#endif
//
//	//1.perform relax pre- iterations with the basic solver
//	sparseMatSorNoResidual(A, X, X, B, w, preIterations);
//
//	//2.coarse grid correction
//	FArray defect_h(h);
//	//compute(Ax-B)
//	mresidual(A,B,X,defect_h);			
//	int size2h = h/2;
//
//	//3.compute X2h -approximation of the X on the next coarser grid
//	FArray X2h(size2h);//size X2h
//	restrictionX(X, X2h);
//
//	//4.compute B2h
//	FArray A2hX2h(size2h);//size temp2h
//	FArray B2h(size2h);//size B2h
//	coarseGridOperator(A, X2h, A2hX2h);
//
//	FArray defect_2h(size2h);
//	restrictionV(defect_h, defect_2h);
//
//	sumVectors(A2hX2h, defect_2h, B2h, 0);
//	//solve A2h*X2h = B2h
//	FArray X2hCopy(X2h.size());
//	CRSSparseMat* A2h = new CRSSparseMat();
//	restrictionM(A, *A2h);
//	//Solve error equation
//
//	if((size2h >= 2) && ((size2h%2) == 0)){
//		for(int i = 0; i < cycleType && currentLevel >= 0; ++i){
//			multigridIteration(currentLevel - 1, *A2h, B2h, w, X2hCopy,preIterations,postIterations,cycleType);
//		}					
//	}
//
//	FArray defectSolutionDifference2h(size2h);
//	FArray defectSolutionDifference(X.size());
//	//defectSolutionDifference = X2hCopy - X2h
//	sumVectors(X2hCopy, X2h, defectSolutionDifference2h,1);
//	for(int i=0; i< size2h	;i++){
//		defectSolutionDifference2h.ptr[i] /= pow((float)DEFECT_FACTOR,currentLevel);
//	}
//	prolongation(defectSolutionDifference2h, defectSolutionDifference);
//
//	//X = X + Difference Of defect
//	sumVectors(X, defectSolutionDifference, X, 0);
//
//	sparseMatSorNoResidual(A, X, X, B, w, postIterations);
//
//	delete A2h;
//}
//
///*********************************************************************************************************/
///*FAS algorithm functions**************************************************************************/
///*********************************************************************************************************/
//
////build coarse grid X2h of Xh using avg of 4 cells, result is saved in X2h(use for coarse X)
//void LinearSolver::restrictionX(const FArray& Xh,FArray& X2h){
//	if(Xh.size()!= 2*X2h.size() ){
//		std::cout << "Error restrictionX size of Xh is 2* size of X2h" << std::endl;
//		exit(-1);
//	}
//	unsigned int i;		
//	for(i=0;i<X2h.size();i++){									
//		float avg = (Xh.ptr[2*i]+ Xh.ptr[2*i+1])/2;
//		X2h.ptr[i] = avg;					
//	}		
//}
////build coarse grid X2h of Xh using sum of 4 cells, result is saved in X2h(use for coarse residual)
//void LinearSolver::restrictionV(const FArray& Xh,FArray& X2h){
//	if(Xh.size()!= 2*X2h.size() ){
//		std::cout << "Error restrictionX size of Xh is 2* size of X2h" << std::endl;
//		exit(-1);
//	}
//	unsigned int i;		
//	for(i=0;i<X2h.size();i++){									
//		float sum = Xh.ptr[2*i]+ Xh.ptr[2*i+1];
//		X2h.ptr[i] = sum;					
//	}	
//
//}
//
//void LinearSolver::restrictionM(CRSSparseMat& Ah, CRSSparseMat& A2h){
//	//if(Ah.dimR()!= 2*A2h.dimR()){
//	//	printf("Error restrictionX size of Xh is 2* size of X2h\n");
//	//	exit(-1);
//	//}
//	COOrdSparseMat* tmp = new COOrdSparseMat(Ah.dimR(), Ah.dimC(), Ah.nonZeros(), new float[Ah.nonZeros()], new int[Ah.nonZeros()], new int[Ah.nonZeros()]);
//	float* tVal = tmp->val();
//	int* tRow = tmp->rowIdx();
//	int* tCol = tmp->colIdx();
//	int* rows = Ah.rowPtr();
//	int endOfRow = 0;
//	float temp;
//	int celCount = 0;
//	for (int row = 0; row < Ah.dimR(); row += 2, rows += 2)
//	{
//		endOfRow = *(rows + 1);
//		bool lastRow = false;
//		if (row == Ah.dimR() - 1) lastRow = true;
//		if (row == 202)
//			row = 202;
//		for (int col = *rows; col < endOfRow; ++col)
//		{
//			float sum;
//			int i = row;
//			int j = Ah.colIdx(col);
//			if (lastRow){
//				if ((j % 2) == 0){
//					sum = Ah(i,j) + Ah(i, j + 1);
//					if (Ah.colIdx(col + 1) == j + 1) ++col;
//				}else{
//					sum = Ah(i,j);
//				}
//			}else{
//				if ((j % 2) == 0){
//					sum = Ah(i,j) + Ah(i, j + 1) + Ah(i + 1, j) + Ah(i + 1, j + 1);
//					if (Ah.colIdx(col + 1) == j + 1) ++col;
//				}
//				else{
//					sum = Ah(i,j) + Ah(i + 1, j);
//				}	
//			}
//			*(tVal++) = sum;
//			*(tRow++) = i / 2;
//			*(tCol++) = j / 2;
//			++celCount;
//		}
//	}
//	A2h.build(*tmp, celCount);
//	delete tmp;
//	//ofstream f;
//	//f.open ("newA2h.txt");
//	//for(int i = 0; i < A2h.rows(); ++i){
//	//	for(int r = A2h.rowPtr(i); r < A2h.rowPtr(i + 1); ++r){
//	//		f << i << ", " << A2h.colIdx(r) << ": " << A2h.val(r) << endl;
//	//	}
//	//}
//	//f.close();
//}
//
////void LinearSolver::restrictionM(CRSSparseMat& Ah,CRSSparseMat& A2h){
////	if(Ah.dimR()!= 2*A2h.dimR() ){
////		std::cout << "Error restrictionX size of Xh is 2* size of X2h" << std::endl;
////		exit(-1);
////	}
////	int i,j;	
////	for (SparseMat<float>::const_row_iter row = Ah.begin(); row != Ah.end(); row++){
////		SparseMat<float>::const_row_iter rowTmp = row;
////		rowTmp++;
////		bool lastRow = false;
////		if(rowTmp == Ah.end()){
////			lastRow = true;
////		}		
////		for( SparseMat<float>::const_col_iter col = row->second.begin(); col != row->second.end(); col++){
////			/*SparseMat<float>::const_col_iter colTmp = col;
////			colTmp++;*/
////			float sum;
////			int i=row->first;
////			int j = col->first;
////			if(lastRow){
////				if((j%2) == 0){//EVEN col
////					sum = Ah(i,j) + Ah(i,j+1);
////					col++;
////				}
////				else{//ODD col
////					sum = Ah(i,j);
////				}
////			}
////			else{//not last row
////				if((j%2) == 0){//EVEN col
////					sum = Ah(i,j) + Ah(i,j+1)+ Ah(i+1,j)+ Ah(i+1,j+1);
////					col++;
////				}
////				else{//ODD col
////					sum = Ah(i,j) + Ah(i+1,j);
////				}					
////			}
////			A2h(i/2,j/2) = sum;
////		}		
////		row = rowTmp;
////	}		
////}
//
//void LinearSolver::prolongation(const FArray& X2h,FArray& Xh){
//	if(Xh.size()!= 2*X2h.size() ){
//		std::cout << "Error prolongation size of Xh is 2* size of X2h" << std::endl;
//		exit(-1);
//	}
//	unsigned int i;		
//	for(i = 0; i < X2h.size(); ++i){									
//		float value = X2h.ptr[i];
//		Xh.ptr[2*i] = value;
//		Xh.ptr[2*i+1] = value;					
//	}	
//}
//
///**1:build fine grid Xh from X2h;
//*  2:apply matrix Ah on Xh save the result in Bh
//*  3:build coarse grid B2h from Bh
//**/
//void LinearSolver::coarseGridOperator(CRSSparseMat& Ah,const FArray& X2h,FArray& B2h){	
//	FArray Xh(2*X2h.size());
//	FArray Bh(2*X2h.size());
//	prolongation(X2h,Xh);
//	Ah.MulVector(Xh.ptr, Bh.ptr);
//	restrictionV(Bh, B2h);	
//}
///*********************************************************************************************************/
///*end of FAS algorithm functions**************************************************************************/
///*********************************************************************************************************/
//
///*********************************************************************************************************/
///*help functions**************************************************************************/
///*********************************************************************************************************/
//void LinearSolver::mresidual(CRSSparseMat& A,const FArray& B,const FArray& X,FArray& residualVec){
//	unsigned int i;
//	A.MulVector(X.ptr, residualVec.ptr);
//	for(i = 0; i < residualVec.size(); ++i){
//		residualVec.ptr[i] = B.ptr[i]- residualVec.ptr[i];
//	}
//}
//
////sign = 0 add, sign = 1 sub
//void LinearSolver::sumVectors(const FArray& vec1,const FArray& vec2,FArray& vecResult,int sign){
//	unsigned int i;
//	for(i = 0; i < vec1.size(); ++i){
//		float temp;
//		if(sign == 0){
//			temp = vec1.ptr[i] + vec2.ptr[i] ;
//		}
//		else{
//			temp = vec1.ptr[i] - vec2.ptr[i] ;
//		}
//		vecResult.ptr[i] = temp;
//	}
//}
//
//
///*********************************************************************************************************/
///*end of help functions**************************************************************************/
///*********************************************************************************************************/
