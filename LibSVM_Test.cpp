// LibSVM_Test.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "CxLibSVM.h"
#include <time.h>
#include <iostream>
using namespace std;

void gen_train_sample(vector<vector<double>>& x, vector<double>& y, long sample_num, long dim, double scale);

void gen_test_sample(vector<double>& x, long sample_num, long dim, double scale);

int _tmain(int argc, _TCHAR* argv[])
{
	//��ʼ��libsvm����
	CxLibSVM    svm;
	svm.init_svm_param(svm.param);

	/*1��׼��ѵ������*/
	vector<vector<double>>    x;    //������
	vector<double>    y;            //��������ǩ��
	gen_train_sample(x, y, 200, 10, 1);

	/*1��������֤*/
	int fold = 10;
	svm.do_cross_validation(x, y, svm.param, fold);

	/*2��ѵ��*/
	svm.train(x, y, svm.param);

	/*3������ģ��*/
	string model_path = "svm_model.txt";
	svm.save_model(model_path);

	/*4������ģ��*/
	svm.load_model(model_path);

	/*5��Ԥ��*/
	//���������������
	vector<double> x_test;
	gen_test_sample(x_test, 200, 10, 1);
	double prob_est;
	//Ԥ��
	double value = svm.predict(x_test, prob_est);

	//��ӡԤ�����͸���
	printf("label:%f, prob:%f", value, prob_est);

	return 0;
}

void gen_train_sample(vector<vector<double>>& x, vector<double>& y, long sample_num, long dim, double scale)
{
	srand((unsigned)time(NULL));//�����
	//�����������������
	for (int i = 0; i < sample_num; i++)
	{
		vector<double> rx;
		for (int j = 0; j < dim; j++)
		{
			rx.push_back(scale*(rand() % dim));
		}
		x.push_back(rx);
		y.push_back(1);
	}

	//��������ĸ�������
	for (int i = 0; i < sample_num; i++)
	{
		vector<double> rx;
		for (int j = 0; j < dim; j++)
		{
			rx.push_back(-scale*(rand() % dim));
		}
		x.push_back(rx);
		y.push_back(2);
	}
}

void gen_test_sample(vector<double>& x, long sample_num, long dim, double scale)
{
	srand((unsigned)time(NULL));//�����
	//�����������������
	for (int j = 0; j < dim; j++)
	{
		x.push_back(-scale*(rand() % dim));
	}
}
