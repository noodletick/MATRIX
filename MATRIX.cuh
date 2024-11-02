// ------------------------------------------------------------------------------------------------//
//	Matthew Jouffray 2024
//	header file containing the matrix class implementation
// ------------------------------------------------------------------------------------------------//
#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <iomanip>
#include "./CUTRIX.cuh"
#include <type_traits>

#ifndef MAT_H
#define MAT_H

template <typename U>
class mat {

private:
	unsigned int n, m;
	std::vector<U> matrix;

public:
	static_assert(std::is_arithmetic<U>::value, "U must be arithmetic");
	// -- constructors --
	mat(std::string, unsigned int, unsigned int);
	mat(const std::string &, U, U, unsigned int, unsigned int);
	mat(const U&, unsigned int, unsigned int);
	mat(std::vector<std::vector<U>>&);
	mat(std::vector<U>&, unsigned int, unsigned int);
	mat(std::vector<U>&);
	mat();
	// -- operators --
	// matrix operations
	mat operator+(const mat&); // addition
	mat operator-(const mat&); // subtraction
	mat operator*(const mat&); // multiplication
	mat operator^(const mat&); // broadcasting/Hadamard product
	mat operator/(const mat&); // dividing matrix by 1D vector by row or column 
	// matrix index
	U& operator()(const unsigned int&, const unsigned int&);
	// matrix comparison
	bool operator==(const mat&);
	// matrix assignment
	void operator=(const mat&);
	// scalar operations
	template<class V>
	friend mat<V> operator*(const V a,const mat<V>& A);
	template<class V>
	friend mat<V> operator*(const mat<V>& A, const V a);
	mat operator/(const U);
	// -- utilities -- 
	mat T(); // transpose
	void print(); // print matrix
	unsigned int rows(); // returns number of rows
	unsigned int cols(); // returns number of column
	mat sum(std::string); // sum along axis
	U sum(); // sum all cells
	U max(); // returns the largest element of the matrix
	U min(); // returns the smallest element of the matrix
	mat xp(); // returns e^x of every matrix elements
};

template <typename U>
mat<U>::mat(std::vector<std::vector<U>>& M) { // contractor which accepts 2D std::vectors for 2D matrix
	m = M.size();
	n = M[0].size();
	this->matrix.resize(m * n);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			this->matrix[i * n + j] = M[i][j];
			
		}
	}
}

template <typename U>
mat<U>::mat(std::vector<U>& A, unsigned int b, unsigned int c) { // contractor which accepts a 1D std::vector for 2D matrix

	this->m = b;
	this->n = c;
	this->matrix = A;

}
template <typename U>
mat<U>::mat(std::string arg, unsigned int M, unsigned int N) { // contractor for initializing matrix as zeros or identity matrix
	// read the argument
	std::vector<U> vec(M * N, 0);
	if (arg == "zeros") { // null matrix
		
		matrix = vec;
		m = M;
		n = N;
	}
	else if (arg == "I") { // identity matrix
		if (N != M) {
			std::cout << "Cannot generate identity rectangular matrix, m must be equal to n." << std::endl;
			vec.clear();
			exit(0);
		}
		m = M;
		n = M;
		
		for (unsigned int i = 0; i < m; i++) {
			for (unsigned int j = 0; j < n; j++) {
				if (i == j) {
					vec[i*n+j] = 1;
				}

			}
		}

		matrix = vec;
	}
	else { // error
		std::cout << "invalid matrix constructor argument" << std::endl;
		vec.clear();
		exit(0);
	}


}
template <typename U>
mat<U>::mat(const std::string &arg, U a, U b, unsigned int M, unsigned int N) { // constructor to initialize matrix with random numbers
	// read the argument

	if (arg == "rand") { 
		std::vector<U> vec(M*N, 0);

		m = M;
		n = N;

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(a, b);

		for (unsigned int i = 0; i < m; i++) {
			for (unsigned int j = 0; j < n; j++) {
				vec[i*n+j] = dis(gen);
			}
		}

		matrix = vec;
	}

	else { // error
		std::cout << "invalid matrix constructor arguments" << std::endl;
		exit(0);
	}


}
template <typename U>
mat<U>::mat(const U& a, unsigned int M, unsigned int N) { // constructor to initialize all values as 'a' for given matrix size

	m = M;
	n = N;
	matrix = vec(M * N, a);

}
template <typename U>
mat<U>::mat(std::vector<U>& M) {
	// contractor which accepts 1D std::vector<U>
	m = M.size();
	n = 1;
	matrix = M;
}
template <typename U>
mat<U>::mat() { //default constructor


	m = 0;
	n = 0;

}

template <typename U>
U& mat<U>::operator()(const unsigned int& i, const unsigned int& j) {
	return this->matrix[i*this->n+j];
}
template <typename U>
void mat<U>::operator=(const mat& B) { // overload of assignment operator
	this->m = B.m;
	this->n = B.n;
	this->matrix = B.matrix;
}
template <typename U>
mat<U> mat<U>::operator*(const mat<U>& B) { // matrix multiplication
	if (this->n != B.m) {
		std::cout << "matrix multiplication error, matrix dimension mismatch.\n";
		exit(0);
	}

	std::vector<U> retrn(this->rows()* B.n);

	int a = this->rows();
	int b = this->cols();
	
	int c = B.m;
	int d = B.n;

	matmult(this->matrix.data(), B.matrix.data(), retrn.data(), a, b, c, d);
	mat<U> mult(retrn, this->m, B.n);
	retrn.clear();
	
	return mult;
}
template <typename U>
bool mat<U>::operator==(const mat& B) { // overlaod of comparison operator
	if (this->matrix == B.matrix) { return true; }
	else { return false; }
}
template <typename U>
mat<U> mat<U>::operator+(const mat<U>& B) { // matrix addition & broadcasting

	mat<U> sum("zeros", this->m, this->n);
	
	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix addition
		
		matop(0, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vector B to match columns
		matop(3, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vector B to match rows
		matop(4, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix addition error, matrix dimension mismatch.\n";
		exit(0);
	}
	
	return sum;
}
template <typename U>
mat<U> mat<U>::operator-(const mat<U>& B) { // matrix subtraction & broadcasting

	mat<U> sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix subtraction error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // classic matrix subtraction
		matop(1, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vector B to match columns
		matop(5, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vector B to match rows
		matop(6, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix subtraction error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}
template <typename U>
mat<U> mat<U>::operator^(const mat<U>& B) { // matrix Hadamard product and broadcasting

	mat<U> sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix broadcasting error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // Hadamard Product
		matop(2, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(7, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(8, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix broadcasting error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}
template <typename U>
mat<U> mat<U>::operator/(const mat<U>& B) { // matrix element by element division and broadcasting

	for (int i = 0; i < B.matrix.size(); i++) {
		if (B.matrix[i] == 0){
			std::cout << "matrix element division error, dividing by zero.\n";
			exit(0);
		}
	}

	mat<U> sum("zeros", this->m, this->n);

	if (this->m != B.m && this->n != B.n) {
		std::cout << "matrix element division error, matrix dimension mismatch.\n";
		exit(0);
	}
	else if (this->m == B.m && this->n == B.n) { // naive division
		matop(9, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->m == B.m && B.n == 1) { // extrude the vetor B to match columns
		matop(10, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else if (this->n == B.n && B.m == 1) {// extrude the vetor B to match rows
		matop(11, this->matrix.data(), B.matrix.data(), sum.matrix.data(), this->m, this->n);
	}
	else {
		std::cout << "matrix element division error, matrix dimension mismatch.\n";
		exit(0);
	}

	return sum;
}
template<class V>
mat<V> operator*(const mat<V>& A, const V a) { // scalar multiplication

	mat<V> sub("zeros", A.m, A.n);

	matscal(0, a, A.matrix.data(), sub.matrix.data(), A.m, A.n);

	return sub;
}
template<class V>
mat<V> operator*(const V a, const mat<V>& A) { // scalar multiplication

	mat<V> sub("zeros", A.m, A.n);
	
	matscal(0, a, A.matrix.data(), sub.matrix.data(), A.m, A.n);

	return sub;
}
template <typename U>
mat<U> mat<U>::operator/(const U a) { // scalar division
	if (a == 0) {
		std::cout << "error: matrix divided by scalar 0.\n";
		exit(0);
	}
	mat<U> sub("zeros", m, n);

	matscal(1, a, matrix.data(), sub.matrix.data(), m, n);

	return sub;
}
template <typename U>
mat<U> mat<U>::sum(std::string arg) { // summing columns or rows
	if (arg == "rows") {
		mat<U> SUM("zeros", this->m, 1);
		U b = 0;
		int a = 3;
		matscal(a, b, matrix.data(), SUM.matrix.data(), m, n);
		return SUM;
	}
	else if (arg == "cols") {
		mat<U> SUM("zeros", 1, this->n);
		U b = 0;
		int a = 2;
		matscal(a, b, matrix.data(), SUM.matrix.data(), m, n);
		return SUM;
	}
	else {
		std::cout << "improper argument for 'sum()' please use 'cols' to sum columns, or 'rows' to sum rows.\n";
		exit(0);
	}


}
template <typename U>
U mat<U>::sum() { // summing all matrix elements into a scalar value
	U sum = 0;
#pragma omp parallel for reduction (+:sum)
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			sum += this->matrix[i*n+j];
		}
	}
	return sum;
}
template <typename U>
U mat<U>::max() {
	if (this->m == 0 && this->n == 0) { return 0; }
	else if (matrix.size() == 1) {
		return this->matrix[0];
	}
	else {
		U max = this->matrix[0];
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				if (max < this->matrix[i * n + j]) {
					max = this->matrix[i * n + j];
				}
			}
		}
		return max;
	}
	

}
template <typename U>
U mat<U>::min() {
	if (this->m == 0 && this->n == 0) { return 0; }
	else if (matrix.size() == 1) {
		return this->matrix[0];
	}
	else {
		U min = this->matrix[0];
		for (int i = 0; i < this->m; i++) {
			for (int j = 0; j < this->n; j++) {
				if (min > this->matrix[i * n + j]) {
					min = this->matrix[i * n + j];
				}
			}
		}
		return min;
	}


}
template <typename U>
mat<U> mat<U>::xp() {
	mat temp("zeros", m, n);
	#pragma omp parallel for
	for (int i = 0; i < this->m; i++) {
		for (int j = 0; j < this->n; j++) {
			temp(i, j) = exp(this->matrix[i * n + j]);
		}
	}
	return temp;
}

template <typename U>
mat<U> mat<U>::T() { // matrix transpose function
	mat<U> transpose("zeros", this->n, this->m); // needs destructor
	mat_indx(this->matrix.data(), transpose.matrix.data(), this->m, this->n);

	return transpose;
}

template <typename U>
unsigned int mat<U>::rows() {
	return this->m;
}
template <typename U>
unsigned int mat<U>::cols() {
	return this->n;
}
template <typename U>
void mat<U>::print() { // matrix print function
	std::cout << std::setprecision(3);
	unsigned int N;
	if (n > 20) {
		std::cout << "the matrix it too wide to print, printing only the first 20 columns: \n";
		N = 20;
	}
	else {
		N = n;
	}

	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < N; j++) {
			if (j == 0) {
				std::cout << "\n| " << matrix[i * N + j] << (j == N - 1 ? " |" : "");
			}
			else {
				std::cout << std::setw(8) << matrix[i * N + j] << (j == N - 1 ? " |" : "");
			}

		}
	}
	std::cout << std::endl<< std::endl;
}

#endif
