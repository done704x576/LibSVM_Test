#ifndef _CXLIBSVM_H_H
#define _CXLIBSVM_H_H

#include <string>
#include <vector>
#include <iostream>
#include "svm.h"

using namespace std;

//ÄÚ´æ·ÖÅä
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class CxLibSVM
{
public:

	struct svm_parameter    param;

private:

	struct svm_model*    model_;

	struct svm_problem        prob;

	struct svm_node *        x_space;

public:

	CxLibSVM();

	~CxLibSVM();

	void init_svm_param(struct svm_parameter& param);

	void train(const vector<vector<double>>&  x, const vector<double>& y, const struct svm_parameter& param);

	int predict(const vector<double>& x,double& prob_est);

	void do_cross_validation(const vector<vector<double>>&  x, const vector<double>& y, const struct svm_parameter& param, const int & nr_fold);

	int load_model(string model_path);

	int save_model(string model_path);

	void free_model();
};



#endif // !_CXLIBSVM_H_H
