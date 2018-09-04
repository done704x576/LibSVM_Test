#include "stdafx.h"
#include "CxLibSVM.h"

CxLibSVM::CxLibSVM()
{
	model_ = NULL;
}

CxLibSVM::~CxLibSVM()
{
	 free_model();
}

void CxLibSVM::init_svm_param(struct svm_parameter& param)
{
	//参数初始化，参数调整部分在这里修改即可
	// 默认参数
	param.svm_type = C_SVC;        //算法类型
	param.kernel_type = LINEAR;    //核函数类型
	param.degree = 3;    //多项式核函数的参数degree
	param.coef0 = 0;    //多项式核函数的参数coef0
	param.gamma = 0.5;    //1/num_features，rbf核函数参数
	param.nu = 0.5;        //nu-svc的参数
	param.C = 10;        //正则项的惩罚系数
	param.eps = 1e-3;    //收敛精度
	param.cache_size = 100;    //求解的内存缓冲 100MB
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 1;    //1表示训练时生成概率模型，0表示训练时不生成概率模型，用于预测样本的所属类别的概率
	param.nr_weight = 0;    //类别权重
	param.weight = NULL;    //样本权重
	param.weight_label = NULL;    //类别权重
}

void CxLibSVM::train(const vector<vector<double>>& x, const vector<double>& y, const struct svm_parameter& param)
{
	if (x.size() == 0)
	{
		return;
	}

	//释放先前的模型
	free_model();

	/*初始化*/        
	long    len = x.size();
	long    dim = x[0].size();
	long    elements = len * dim;

	//转换数据为libsvm格式
	prob.l = len;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements + len);
	int j = 0;
	for (int l = 0; l < len; l++)
	{
		prob.x[l] = &x_space[j];
		for (int d = 0; d < dim; d++)
		{                
			x_space[j].index = d+1;
			x_space[j].value = x[l][d];    
			j++;
		}
		x_space[j++].index = -1;
		prob.y[l] = y[l];
	}

	/*训练*/
	model_ = svm_train(&prob, &param);    
}

int CxLibSVM::predict(const vector<double>& x,double& prob_est)
{
	//数据转换
	svm_node* x_test = Malloc(struct svm_node, x.size()+1);
	for (unsigned int i=0; i<x.size(); i++)
	{
		x_test[i].index = i + 1;
		x_test[i].value = x[i];
	}
	x_test[x.size()].index = -1;
	double *probs = new double[model_->nr_class];//存储了所有类别的概率
	//预测类别和概率
	int value = (int)svm_predict_probability(model_, x_test, probs);
	for (int k = 0; k < model_->nr_class;k++)
	{//查找类别相对应的概率
		if (model_->label[k] == value)
		{
			prob_est = probs[k];
			break;
		}
	}
	delete[] probs;
	return value;
}

void CxLibSVM::do_cross_validation(const vector<vector<double>>& x, const vector<double>& y, const struct svm_parameter& param, const int & nr_fold)
{
	if (x.size() == 0)return;

	/*初始化*/
	long    len = x.size();
	long    dim = x[0].size();
	long    elements = len*dim;

	//转换数据为libsvm格式
	prob.l = len;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements + len);
	int j = 0;
	for (int l = 0; l < len; l++)
	{
		prob.x[l] = &x_space[j];
		for (int d = 0; d < dim; d++)
		{
			x_space[j].index = d + 1;
			x_space[j].value = x[l][d];
			j++;
		}
		x_space[j++].index = -1;
		prob.y[l] = y[l];
	}

	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);

	svm_cross_validation(&prob, &param, nr_fold, target);
	if (param.svm_type == EPSILON_SVR ||
		param.svm_type == NU_SVR)
	{
		for (i = 0; i < prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y)*(v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy - sumv*sumy)*(prob.l*sumvy - sumv*sumy)) /
			((prob.l*sumvv - sumv*sumv)*(prob.l*sumyy - sumy*sumy))
			);
	}
	else
	{
		for (i = 0; i < prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.l);
	}
	free(target);
}

int CxLibSVM::load_model(string model_path)
{
	//释放原来的模型
	free_model();
	//导入模型
	model_ = svm_load_model(model_path.c_str());
	if (model_ == NULL)return -1;
	return 0;
}

int CxLibSVM::save_model(string model_path)
{
	int flag = svm_save_model(model_path.c_str(), model_);
	return flag;
}

void CxLibSVM::free_model()
{
	if (model_ != NULL)
	{
		svm_free_and_destroy_model(&model_);
		svm_destroy_param(&param);

		if (prob.y != NULL)
		{
			free(prob.y);
			prob.y = NULL;
		}
		
		if (prob.x != NULL)
		{
			free(prob.x);
			prob.x = NULL;
		}
		
		if (x_space != NULL)
		{
			free(x_space);
			x_space = NULL;
		}
	}
}

