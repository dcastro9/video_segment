#pragma once

class COOrdSparseMat {
	private:
		float* _val;
		int* _rowIdx;
		int* _colIdx;

		int _nz;
		int _rows;
		int _cols;

	public:
		COOrdSparseMat(int M, int N, int nz, float* val, int* row, int* col);
		
		~COOrdSparseMat(void);

		// It is not recommended that you use the three following functions unless
		// you know what you are doing.
		// TODO(dcastro): May be best to remove these.
		float* val() {return _val;}
		int* rowIdx() {return _rowIdx;}
		int* colIdx() {return _colIdx;}

		float& val(int i) {return _val[i];}
		int& rowIdx(int i) {return _rowIdx[i];}
		int& colIdx(int i) {return _colIdx[i];}

		const float& val(int i) const {return _val[i];}
		const int& rowIdx(int i) const {return _rowIdx[i];}
		const int& colIdx(int i) const {return _colIdx[i];}

		int rows() const {return _rows;}
		int cols() const {return _cols;}
		int nonZeros() const {return _nz;}

		COOrdSparseMat& operator = (const COOrdSparseMat& other);

		float operator() (int i, int j) const;
		void set(int i, int j, float val);

		friend class CRSSparseMat;
};

