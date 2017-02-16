"""
This is module which designed to predict enrollment


Created on 2016/12/03
Version: 1.0
@author: Sheng Liu
ShengLiu Copyright 2016-2017
Please report any bugs to shengliu [at] nyu [dot]edu
"""
from sklearn import preprocessing
import pandas as pd
import numpy as np
import warnings
from sklearn import preprocessing
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn import datasets, linear_model
import scipy
warnings.filterwarnings('ignore')

class Markov_Enrollment_Initiate:
	def __init__(self,data,n_year):
		self.data = data
		self.k_data = data[data['Grade Level'] == 'K']['Count of Students'].values
		self.n_year = n_year

	@staticmethod
	def calc_cumu_sum(data):
		n = len(data)
		return [sum(data[0:i+1]) for i in range(n)]



	def Train_Kindergarten(self,start_year,end_year):
		#min_max_scaler = preprocessing.MinMaxScaler()
		data_train = self.k_data[start_year:end_year]
		data_train_agg = Markov_Enrollment_Initiate.calc_cumu_sum(data_train)
		X = data_train_agg[0:-1]
		#X_2 = data_train_agg[0:-2]
		#X_2 = data_train[0:-1]
		#X = [[X[i],X_2[i]] for i in range(len(X))]
		X = [[X[i]] for i in range(len(X))]
		#X_minmax = min_max_scaler.fit_transform(X)
		#print(X)
		Y = data_train_agg[1:]
		regr = linear_model.LinearRegression()
		regr.fit(X, Y)
		
		return regr

	def Test_Kindergarten(self,regr,Y_test,X_test):
		#Calc Meadian of Relative Error
		err = np.median((regr.predict(X_test)-Y_test)/Y_test)
		plt.scatter(X_test, Y_test,  color='black')
		plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
		plt.title('Performance of The Model on Test Set')
		plt.xticks(())
		plt.yticks(())
		plt.show()
		return err

	def Predict_Kindergarten(self,regr,agg):
		return (regr.predict(agg) - agg)[0]
	#def Predict_Kindergarten(self,regr,agg):
	#	return (regr.predict(agg) - agg[0])[0]

class Markov_Enrollment_Transit:

	def __init__(self,data,K_IN_THIS_YEAR):
		self.data = data
		self.k_data = data[data['Grade Level'] == 'K']
		self.K_IN_THIS_YEAR = K_IN_THIS_YEAR

	@staticmethod
	def Estimate_Transit_Matrix(Students_in_Grade):
		A = np.zeros([5,6])
		A[0][0] = Students_in_Grade[0]
		#A[0][1] = Students_in_Grade[1]
		A[1][1] = Students_in_Grade[1]
		#A[1][2] = Students_in_Grade[2]
		A[2][2] = Students_in_Grade[2]
		#A[2][3] = Students_in_Grade[3] 
		A[3][3] = Students_in_Grade[3] 
		#A[3][4] = Students_in_Grade[4]
		A[4][4] = Students_in_Grade[4]
		A[:,5] = Students_in_Grade[0]*0
		#A[:,5] = 50
		return A




	def Estimate_Transit_Matrix_dic(self):
		data_in_tract = self.data
		A_dict = {}
		B_dict = {}
		#test = [40, 70, 50, 30, 70, 40, 80, 60, 60, 40, 40, 40, 40, 40]
		for year in np.unique(data_in_tract['Year']):
			data_in_tract_this_year = data_in_tract[data_in_tract['Year'] == year]
			Students_in_Grade = data_in_tract_this_year['Count of Students'].values
			data_in_tract_next_year = data_in_tract[data_in_tract['Year'] == year+1]
			#K = data_in_tract_this_year[data_in_tract_this_year['Grade Level'] == 'K']['Count of Students'].values
			A = Markov_Enrollment_Transit.Estimate_Transit_Matrix(Students_in_Grade)
			if np.size(data_in_tract_next_year) > 0:
				B = np.zeros(5)
				B = data_in_tract_next_year['Count of Students'].values[1:] #+ data_in_tract_this_year['Count of Students'].values[1:]
				B_dict[int(year-2000)] = B
			A_dict[int(year-2000)] = A
		return A_dict,B_dict
	@staticmethod
	def Estimate_Transit_Matrix_dic_new(A_dict,B_dict,predict_Other_Grade,year):
		A_dict[year] = Markov_Enrollment_Transit.Estimate_Transit_Matrix(predict_Other_Grade)
		B_dict[year] = np.array(predict_Other_Grade[1:])
		return A_dict,B_dict

	


	@staticmethod
	def train_Grade_One_to_Five(A_dict, B_dict,start,end):
		A_sum = np.zeros([6,6])
		bA_sum =np.zeros(6)

		for key in range(start,end):
			A_sum += np.dot(A_dict[key].T, A_dict[key])

		for key in range(start,end,1):
			bA_sum = bA_sum + np.dot(B_dict[key].T,A_dict[key]) 
		m = matrix(A_sum)
		n = matrix(bA_sum)
		#G = matrix(np.r_[np.identity(6).tolist(),((-1*np.identity(6)).tolist())])
		#h = matrix(np.r_[1*np.ones(5),3,0*np.ones(5),0])
		G = matrix(((-1*np.identity(6)).tolist()))
		h = matrix(0*np.ones(6))
		sol=solvers.qp(m, -n,G,h)
		#sol = scipy.optimize.nnls(A_sum,bA_sum)
		print(np.dot(A_sum,sol['x']).T-bA_sum)
		print('The solution to this problem is \n',sol['x']) 
		#print('The solution to this problem is \n',sol) 
		return np.array(sol['x']) # np.array(sol[0])
	


	@staticmethod
	def predict_Grade_One_to_Five(A,sol):

		prediction = np.dot(A,sol).T 
		#print('123',prediction)
		return prediction



if __name__ == '__main__':
	data = pd.read_csv('../CSD20_Resident_Data_Phase_1.csv', header = 0, sep = ',')#, index_col = 0)
	data['Year'] = np.round(data['School Year']/10000)
	for tract in np.unique(data['2010 Census Tract']):
		data_in_tract = data[data['2010 Census Tract']== tract]
		model = Markov_Enrollment_Initiate(data_in_tract,5).Train_Kindergarten()


