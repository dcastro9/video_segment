#pragma once
#define NULL 0

class FArray{
private:
	size_t _size;
	bool _autoClean;
public:
	
	float* ptr;
	
	FArray():_size(0), _autoClean(false), ptr(NULL){}

	FArray(const size_t s): _size(s){
		ptr = new float[s];
	}

	FArray(const size_t s, float* p, bool autoClean = true): _size(s), ptr(p), _autoClean(autoClean){
	}

	FArray(const size_t s, const bool init, const float initVal = 0): _size(s), ptr(new float[s]){
		if (init){
			for (unsigned int i = 0; i < s; ++i)
				ptr[i] = initVal;
		}
	}

	inline size_t size() const{
		return _size;
	}

	inline void set(float to){
		for (unsigned int i = 0; i < _size; ++i)
				ptr[i] = to;
	}

	void clean()
	{
		if (ptr != NULL){
			delete[] ptr;
			ptr = NULL;
		}
	}

	~FArray(){
		if (ptr != NULL && _autoClean)
			delete[] ptr;
	}
};