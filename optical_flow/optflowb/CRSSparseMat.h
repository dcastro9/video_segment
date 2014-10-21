#pragma once
#include "COOrdSparseMat.h"
class CRSSparseMat
{
	private:
		float* _val;
		int* _rowPtr;
		int* _colIdx;

		int _nz;
		int _rows;
		int _cols;

	public:
		CRSSparseMat();
		//CRSSparseMat(const COOrdSparseMat& s);
		//CRSSparseMat(int M, int N, int nz, float* val, int* r, int* c);

		~CRSSparseMat(void);

		//BOAZ: UNSAFE TO USE, you better know what you're doing.
		float* val() {return _val;}
		//BOAZ: UNSAFE TO USE, you better know what you're doing.
		int* rowPtr() {return _rowPtr;}
		//BOAZ: UNSAFE TO USE, you better know what you're doing.
		int* colIdx() {return _colIdx;}

		float& val(int i) {return _val[i];}
		int& rowPtr(int i) {return _rowPtr[i];}
		int& colIdx(int i) {return _colIdx[i];}

		const float& val(int i) const {return _val[i];}
		const int& rowPtr(int i) const {return _rowPtr[i];}
		const int& colIdx(int i) const {return _colIdx[i];}

		int dimR() const {return _rows;}
		int dimC() const {return _cols;}
		int nonZeros() const {return _nz;}

		void build (const COOrdSparseMat& s);
		void build(const COOrdSparseMat& s, int nz);
		void MulVector(const float* x, float* b);
		void MulScalar(const float x);

		CRSSparseMat& operator= (const CRSSparseMat& other);

		//float* operator() (int i, int j);
		float operator()(int i, int j)  const;
		//float value(int i, int j) const;
		//void set(int i, int j, float val);
};

