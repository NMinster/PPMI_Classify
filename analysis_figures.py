import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost.sklearn import XGBRegressor  

from sklearn.model_selection import train_test_split
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from sklearn.metrics import recall_score, precision_score, classification_report,accuracy_score,confusion_matrix, roc_curve, auc, roc_curve,accuracy_score,plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, normalize
from scipy import ndimage
import seaborn as sns

import os
for dirname, _, filenames in os.walk('~/YOUR_DIR_HERE'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from scipy.stats import uniform, truncnorm, randint
        
        
    
#Random forest
#Hyperparameters search grid 

model_params = {
    'n_estimators': randint(4,200),
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    'min_samples_split': uniform(0.01, 0.199)
}

rf_model = RandomForestClassifier()
# Create the GridSearchCV object
rf_search =  RandomizedSearchCV(rf_model, model_params, n_iter=100, cv=5, random_state=1, n_jobs = -1, verbose=2)
rf_search.fit(x_train_ov, y_train_ov)

best_accuracy = rf_search.best_score_ 
best_parameters = rf_search.best_params_
best_rf = rf_search.best_estimator_
best_rf

#Use best parameters
rf_model = RandomForestClassifier(bootstrap=False, max_features=0.25411986663039127, min_samples_leaf=8,
                       min_samples_split=0.013158723180134153, n_estimators=116)

rf_model.fit(x_train_ov,y_train_ov)

prediction=rf_model.predict(x_test)

acc_random_forest = accuracy_score(prediction,y_test)
print('Validation accuracy of RandomForest Classifier is', acc_random_forest)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cmap="Greens",fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = rf_model.predict_proba(x_test)[:,1]
fpr1,tpr1,thresholds1 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)

#Analysis of model 
IDs= degs.columns
IDs= IDs.drop(['pd'])
feature_scores = pd.Series(rf_model.feature_importances_, index=IDs).sort_values(ascending=False)


#SVM
#Hyperparameters search grid 
svc_params = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

search = RandomizedSearchCV(estimator = SVC(C=2.3), param_distributions = svc_params, n_iter = 2, cv = 2, verbose=2, random_state=100 , n_jobs = -1)

search.fit(x_train_ov, y_train_ov)

best_accuracy = search.best_score_ 
best_parameters = search.best_params_

best_svc = search.best_estimator_
best_svc

#build SVM model with best parameters
svc_model = SVC(C=1, kernel='linear', gamma=0.38 ,probability=True)

svc_model.fit(x_train_ov, y_train_ov)

prediction=svc_model.predict(x_test)

acc_svc = accuracy_score(prediction,y_test)
print('The accuracy of SVM is', acc_svc)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
ax = plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True, cmap='Greens', fmt = "d",linecolor="k",linewidths=3)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
ax.xaxis.set_ticklabels(['HC','PD'])
ax.yaxis.set_ticklabels(['HC','PD'], rotation=0)
plt.title("SVM CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = svc_model.predict_proba(x_test)[:,1]
fpr2,tpr2,thresholds2 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ('Area under the curve :',(auc(fpr,tpr))),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = 'lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("SVM ROC - CURVE & AREA UNDER CURVE",fontsize=20)


#KNN
#Grid search
knn_param = {
    "n_neighbors": [i for i in range(1,30,5)],
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": [1, 10, 30],
    "p": [1,2]
}
search = GridSearchCV(KNeighborsClassifier(), knn_param, n_jobs=-1, verbose=1)
search.fit(x_train_ov, y_train_ov)

best_accuracy = search.best_score_ 
best_parameters = search.best_params_ 
# select best svc
best_knn = search.best_estimator_
best_knn

knn_model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=1, n_neighbors=1, p=1, weights='distance')

knn_model.fit(x_train_ov,y_train_ov)
prediction=knn_model.predict(x_test)

acc_knn = accuracy_score(prediction,y_test)
print('The accuracy of K-NN is', acc_knn)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True, cmap='Greens', fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = knn_model.predict_proba(x_test)[:,1]
fpr3,tpr3,thresholds3 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)

#Logistic Regression
#Grid search
log_grid = {'C': [1e-03, 1e-2, 1e-1, 1, 10], 
                 'penalty': ['l1', 'l2']}

log_model = GridSearchCV(estimator=LogisticRegression(solver='liblinear'), 
                  param_grid=log_grid, 
                  cv=3,
                  scoring='accuracy')
log_model.fit(x_train_ov, y_train_ov)


best_accuracy = log_model.best_score_ 
best_parameters = log_model.best_params_ 

best_lr = log_model.best_estimator_
best_lr

lr_model = LogisticRegression(C=5000, penalty='l1', solver='liblinear')

lr_model.fit(x_train_ov,y_train_ov)

prediction=lr_model.predict(x_test)

acc_log = accuracy_score(prediction,y_test)
print('Validation accuracy of Logistic Regression is', acc_log)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cmap="Greens",fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = lr_model.predict_proba(x_test)[:,1]
fpr4,tpr4,thresholds4 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)

#Decision Tree
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4, 5, 6], 'max_depth':[3,4,5,6,7,8]}
decision_search = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1, verbose=2, cv=3)

decision_search.fit(x_train_ov, y_train_ov)


best_accuracy = decision_search.best_score_ 
best_parameters = decision_search.best_params_ 

best_ds = decision_search.best_estimator_
best_ds

ds_model = DecisionTreeClassifier(max_depth=8, max_leaf_nodes=90, random_state=42,  min_samples_split=5)

ds_model.fit(x_train_ov,y_train_ov)

prediction=ds_model.predict(x_test)

acc_decision_tree = accuracy_score(prediction,y_test)
print('Validation accuracy of Decision Tree is', acc_decision_tree)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cmap="Greens",fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = ds_model.predict_proba(x_test)[:,1]
fpr5,tpr5,thresholds5 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)


#XGBoost
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgbreg = XGBRegressor() 

xgb_search = RandomizedSearchCV(xgbreg, params, n_jobs=-1)



xgb_search.fit(x_train_ov, y_train_ov)

best_accuracy = xgb_search.best_score_ #to get best score
best_parameters = xgb_search.best_params_ #to get best parameters
# select best svc
best_xgb = xgb_search.best_estimator_
best_xgb

xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0, gamma=0.8848547994123379, gpu_id=-1,
              importance_type=None, enable_categorical=False, interaction_constraints='',
              learning_rate=0.2792732351882348, max_delta_step=0, max_depth=21, use_label_encoder=False, 
              min_child_weight=13.710761420156429, monotone_constraints='()',
              n_estimators=26, n_jobs=24, num_parallel_tree=1, random_state=0,
              reg_alpha=11.769827280823089, reg_lambda=1, scale_pos_weight=1, subsample=0.9206972108841006,
              tree_method='exact', validate_parameters=1, verbosity=None)

xgb_model.fit(x_train_ov,y_train_ov)

prediction=xgb_model.predict(x_test)

acc_xgb = accuracy_score(prediction,y_test)
print('Validation accuracy of XG Boost is', acc_xgb)
print ("\nClassification report :\n",(classification_report(y_test,prediction)))

#Confusion matrix
plt.figure(figsize=(13,10))
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,prediction),annot=True,cmap="Greens",fmt = "d",linecolor="k",linewidths=3)
plt.title("CONFUSION MATRIX",fontsize=20)

#ROC curve and Area under the curve plotting
predicting_probabilites = xgb_model.predict_proba(x_test)[:,1]
fpr6,tpr6,thresholds6 = roc_curve(y_test,predicting_probabilites)
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")
plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)

#Summary reports
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Decision Tree',
              'Random Forest', 'XG Boost'],

    'Score': [acc_svc, acc_knn, acc_log, acc_decision_tree, 
              acc_random_forest, acc_xgb]})
models.sort_values(by='Score', ascending=False)

#ROC curve and Area under the curve plotting
plt.plot(fpr2,tpr2,label = ("SVM :", round(auc(fpr2,tpr2),2)),color = "g")
plt.plot(fpr4,tpr4,label = ("LR :",round(auc(fpr4,tpr4),2)),color = "y")
plt.plot(fpr1,tpr1,label = ("RF :",round(auc(fpr1,tpr1),2)),color = "r")
plt.plot(fpr3,tpr3,label = ("KNN :",round(auc(fpr3,tpr3),2)),color = "b")
#plt.plot(fpr7,tpr7,label = ("NB :",round(auc(fpr7,tpr7),2)),color = "k")
plt.plot(fpr6,tpr6,label = ("XGB :",round(auc(fpr6,tpr6),2)),color = "m")
plt.plot(fpr5,tpr5,label = ("DT :",round(auc(fpr5,tpr5),2)),color = "c")
plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")

plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20)
plt.savefig('books_read.png')
plt.show()


#Pickle results and re-run removing the Age at Consent column
import pickle

#with open('train.svm', 'wb') as f:
   # pickle.dump([fpr2, tpr2], f)
    
with open('train.LR', 'wb') as f:
    pickle.dump([fpr4, tpr4], f)
    
with open('train.RF', 'wb') as f:
    pickle.dump([fpr1, tpr1], f)
    
with open('train.KNN', 'wb') as f:
    pickle.dump([fpr3, tpr3], f)
    
with open('train.XGB', 'wb') as f:
    pickle.dump([fpr6, tpr6], f)
    
with open('train.DT', 'wb') as f:
    pickle.dump([fpr5, tpr5], f)
    

#ROC curve and Area under the curve plotting for all models 
    
fig, axs = plt.subplots(2, 3, figsize=(10,7))

fig.suptitle("ROC - CURVE & AREA UNDER CURVE WITH AND WITHOUT AGE DATA")

axs[0, 0].plot(fpr1,tpr1,label = ("w/o :",round(auc(fpr1,tpr1),2)),color = "r")
axs[0, 0].plot(fpr11,tpr11,label = ("w/ :",round(auc(fpr11,tpr11),2)),color = "b")
axs[0, 0].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[0, 0].set_title('RF')
axs[0, 0].legend(loc = "lower right")

axs[0, 1].plot(fpr2,tpr2,label = ("w/o :",round(auc(fpr2,tpr2),2)),color = "r")
axs[0, 1].plot(fpr12,tpr12,label = ("w/ :",round(auc(fpr12,tpr12),2)),color = "b")
axs[0, 1].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[0, 1].set_title('SVM')
axs[0, 1].legend(loc = "lower right")

axs[0, 2].plot(fpr3,tpr3,label = ("w/o :",round(auc(fpr3,tpr3),2)),color = "r")
axs[0, 2].plot(fpr13,tpr13,label = ("w/ :",round(auc(fpr13,tpr13),2)),color = "b")
axs[0, 2].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[0, 2].set_title('KNN')
axs[0, 2].legend(loc = "lower right")

axs[1, 0].plot(fpr4,tpr4,label = ("w/o :",round(auc(fpr4,tpr4),2)),color = "r")
axs[1, 0].plot(fpr14,tpr14,label = ("w/ :",round(auc(fpr14,tpr14),2)),color = "b")
axs[1, 0].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[1, 0].set_title('LR')
axs[1, 0].legend(loc = "lower right")

axs[1, 1].plot(fpr5,tpr5,label = ("w/o :",round(auc(fpr5,tpr5),2)),color = "r")
axs[1, 1].plot(fpr15,tpr15,label = ("w/ :",round(auc(fpr15,tpr15),2)),color = "b")
axs[1, 1].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[1, 1].set_title('DT')
axs[1, 1].legend(loc = "lower right")

axs[1, 2].plot(fpr6,tpr6,label = ("w/o :",round(auc(fpr6,tpr6),2)),color = "r")
axs[1, 2].plot(fpr16,tpr16,label = ("w/ :",round(auc(fpr16,tpr16),2)),color = "b")
axs[1, 2].plot([1,0],[1,0],linestyle = "dashed",color ="k")
axs[1, 2].set_title('XGB')
axs[1, 2].legend(loc = "lower right")


for ax in axs.flat:
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
