//
// Created by matthewjouffray on 10/12/24.
//
#include "MATRIX.cuh"
#include <vector>
#include <string>

int main(){

    //creating matrices using vectors
    std::vector<float> v1, v2, v3, v4, v5, v6;
    v1 = {1.2,2.1,3.0,4.5};
    v2 = {5.2,6.2,7.3,8.7};
    v3 = {1.1,1.2,1.3,1.4};
    v4 = {2.9,2.7,2.4,2.1};
    v5 = {3.3,3.5,3.8,3.0};
    v6 = {4.4,4.6,4.2,4.1};

    std::vector<std::vector<float>> matrix1, matrix2, matrix3, matrix4;
    matrix1 = {v1,v2,v3};
    matrix2 = {v4,v5,v6};
    matrix3 = {v1, v3};
    matrix4 = {v2,v4};

    //creating matrix objects using MATRIX.cuh and the vectors
    mat<float> mat1(matrix1);
    mat<float> mat2(matrix2);
    mat<float> mat3(matrix3);
    mat<float> mat4(matrix4);
    mat<float> mat5, mat6;

    //transposing mat3 if needed for matrix multiplication with mat4
    mat3 = mat3.T();

    //testing the matrix print
    std::cout<<"mat1"<<std::endl;
    mat1.print();
    std::cout<<"mat2"<<std::endl;
    mat2.print();
    std::cout<<"mat3"<<std::endl;
    mat3.print();
    std::cout<<"mat4"<<std::endl;
    mat4.print();

    //summing cols and rows of mat1 and mat2 into mat5 and 6 and printing the result
    mat5 = mat1.sum("rows");
    mat6 = mat2.sum("cols");
    std::cout<<"mat5 is mat1.sum(rows):";
    mat5.print();
    std::cout<<"mat6 is mat2.sum(cols):";
    mat6.print();

    mat3.sum("cols").print();
    std::cout<<"transposing mat2: "<<std::endl;
    mat2.T().print();

    mat6 = mat2 * mat3;
    std::cout<<"mat2 times mat3: "<<std::endl;
    mat6.T().print();
    float tim = 1.2;
    mat6 = tim * mat6;
    std::cout<<"mat times 1.2: "<<std::endl;
    mat6.T().print();
    mat<int> mat7;
    /*mat bigtest("rand",0.016,0.025,10,60000);
    //mat bigtest("rand",0.016,0.025,6,10);
    std::cout<<"bigtest:\n";
    bigtest.print();
    bigtest=bigtest.sum("cols");
    bigtest.print();*/
    return 0;
  }