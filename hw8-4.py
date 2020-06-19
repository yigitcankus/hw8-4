import pandas as pd
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import SVC



#proje 2 ABD ev fiyatları
# Regresyon
# print("Regresyon örneği: ABD Ev fiyatları")
# df = pd.read_csv("final_dataa.csv")
#
#
# df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
# df["zindexvalue"]= df["zindexvalue"].astype(np.int64)
#
# # "bathrooms", "bedrooms","finishedsqft","totalrooms","yearbuilt","zestimate","zindexvalue"
# X = df[["bathrooms","finishedsqft"]]
# y = df.lastsoldprice
#
#
# X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
#
#
# svr_reg = SVR(kernel='linear', C=1)
# svr_reg.fit(X_eğitim, y_eğitim)
#
# y_tahmin = svr_reg.predict(X_test)
# rmse_test = MSE(y_test, y_tahmin)**(1/2)
# print("RMSE değeri (Kernel = linear): {:.2f}".format(rmse_test))
#
#
#
##################################################################################################################
# kernel_listesi = ['linear', 'rbf', 'poly']
#
# for kernel in (kernel_listesi):
#     svr_kernel = SVR(kernel=kernel,C=1)
#     svr_kernel.fit(X_eğitim, y_eğitim)
#     y_tahmin = svr_kernel.predict(X_test)
#     rmse_test = MSE(y_test, y_tahmin) ** (1 / 2)
#     print("RMSE değeri (svr=> kernel = {0}): {1}".format(kernel,rmse_test))
#
# print()
##################################################################################################################
#
# gamma_listesi = [0.1, 1, 10, 100]
#
# for  gamma in (gamma_listesi):
#     svr_gamma = SVR(kernel='rbf', gamma=gamma).fit(X_eğitim, y_eğitim)
#     y_tahmin = svr_gamma.predict(X_test)
#     rmse_test = MSE(y_test, y_tahmin) ** (1 / 2)
#     print("RMSE değeri (svr=> kernel = rbf) (gamma degeri = {0}) : {1}".format(gamma, rmse_test))
#
# print()
##################################################################################################################
#
#
# c_listesi = [0.1, 1, 10, 100, 1000, 10000]
#
# for  c in (c_listesi):
#     svr_c = SVR(kernel='rbf', C=c).fit(X_eğitim, y_eğitim)
#     y_tahmin = svr_c.predict(X_test)
#     rmse_test = MSE(y_test, y_tahmin) ** (1 / 2)
#     print("RMSE değeri (svr=> kernel = rbf) (c degeri = {0}) : {1}".format(c, rmse_test))
#
# print()
#
##################################################################################################################
#
# degree_listesi = [1, 2, 3, 4]
#
# for degree in (degree_listesi):
#     svr_degree = SVR(kernel='poly', degree=degree).fit(X_eğitim, y_eğitim)
#     y_tahmin = svr_degree.predict(X_test)
#     rmse_test = MSE(y_test, y_tahmin) ** (1 / 2)
#     print("RMSE değeri (svr=> kernel = poly) (degree = {0}) : {1}".format(degree, rmse_test))
#
#
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
#
# Proje 3 Fraud credit card
# Classification

# df = pd.read_csv("creditcard_azaltılmış.csv")
#
# X = df.drop('Class', axis=1)
# y = df['Class']
#
# X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=112)
#
# print("Classification örneği, Fraud operations.")
#
# svc = SVC(kernel='linear', C=1)
# svc.fit(X_eğitim, y_eğitim)
# y_tahmin = svc.predict(X_test)
# print("SVM ile Doğruluk Değeri : {:.2f}".format(accuracy_score(y_test, y_tahmin)))
#
# print()
# ##################################################################################################################
#
# kernel_listesi = ['linear', 'rbf', 'poly']
#
# for kernel in (kernel_listesi):
#     svc_kernel = SVC(kernel=kernel,C=1)
#     svc_kernel.fit(X_eğitim, y_eğitim)
#     y_tahmin = svc_kernel.predict(X_test)
#     acc_score = accuracy_score(y_test, y_tahmin)
#     print("Accuracy score değeri (svr=> kernel = {0}): {1}".format(kernel,acc_score))
#
# print()
# ##################################################################################################################
#
# gamma_listesi = [0.1, 1, 10, 100]
#
# for  gamma in (gamma_listesi):
#     svc_gamma = SVC(kernel='rbf', gamma=gamma).fit(X_eğitim, y_eğitim)
#     y_tahmin = svc_gamma.predict(X_test)
#     acc_score = accuracy_score(y_test, y_tahmin)
#     print("Accuracy score değeri (svr=> kernel = rbg) (gamma değeri ={0}): {1}".format(gamma, acc_score))
#
# print()
# ##################################################################################################################
#
# c_listesi = [0.1, 1, 10, 100, 1000, 10000]
#
# for  c in (c_listesi):
#     svc_c = SVC(kernel='rbf', C=c).fit(X_eğitim, y_eğitim)
#     y_tahmin = svc_c.predict(X_test)
#     acc_score = accuracy_score(y_test, y_tahmin)
#     print("Accuracy score değeri (svr=> kernel = rbg) (c değeri ={0}): {1}".format(c, acc_score))
#
# print()
# ##################################################################################################################
#
# degree_listesi = [1, 2, 3, 4]
#
# for degree in (degree_listesi):
#     svc_degree = SVC(kernel='poly', degree=degree).fit(X_eğitim, y_eğitim)
#     y_tahmin = svc_degree.predict(X_test)
#     acc_score = accuracy_score(y_test, y_tahmin)
#     print("Accuracy score değeri (svr=> kernel = poly) (degree ={0}): {1}".format(degree, acc_score))








