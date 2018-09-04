// LibSVM_Test.cpp : 定义控制台应用程序的入口点。
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
	//初始化libsvm对象
	CxLibSVM    svm;
	svm.init_svm_param(svm.param);

	/*1、准备训练数据*/
	vector<vector<double>>    x;    //样本集
	vector<double>    y;            //样本类别标签集
	gen_train_sample(x, y, 200, 10, 1);

	/*1、交叉验证*/
	int fold = 10;
	svm.do_cross_validation(x, y, svm.param, fold);

	/*2、训练*/
	svm.train(x, y, svm.param);

	/*3、保存模型*/
	string model_path = "svm_model.txt";
	svm.save_model(model_path);

	/*4、导入模型*/
	svm.load_model(model_path);

	/*5、预测*/
	//生成随机测试数据
	vector<double> x_test;
	gen_test_sample(x_test, 200, 10, 1);
	double prob_est;
	//预测
	double value = svm.predict(x_test, prob_est);

	//打印预测类别和概率
	printf("label:%f, prob:%f", value, prob_est);

	return 0;
}

void gen_train_sample(vector<vector<double>>& x, vector<double>& y, long sample_num, long dim, double scale)
{
	srand((unsigned)time(NULL));//随机数
	//生成随机的正类样本
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

	//生成随机的负类样本
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
	srand((unsigned)time(NULL));//随机数
	//生成随机的正类样本
	for (int j = 0; j < dim; j++)
	{
		x.push_back(-scale*(rand() % dim));
	}
}
