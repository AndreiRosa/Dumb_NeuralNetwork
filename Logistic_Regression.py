import pandas as pd                                     # Used to treat dataframes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

dfPoints = pd.read_csv("df_points.txt", delimiter="\t") # Import dataset

dfPoints = dfPoints.drop(['Unnamed: 0'], axis=1)        # Remove the ID column

# Divide columns from dataset in dependent and independent variables
independent_variables = dfPoints[['x', 'y', 'z']]
dependent_variable = dfPoints['label']

x = dfPoints['x'].to_numpy()
y = dfPoints['label'].to_numpy()

print(independent_variables.corr(method='pearson'))



# # Divide dataset in training and test samples
# X_train,X_test,y_train,y_test = train_test_split(independent_variables,dependent_variable,test_size=0.10,random_state=0)
#
# # Applies logistic regression
# logistic_regression = LogisticRegression(solver='lbfgs')
# logistic_regression.fit(X_train,y_train)
# y_pred=logistic_regression.predict(X_test)
#
# # Print confusion matrix
# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)
#
# # Show model accuracy
# print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
#
# plt.show()
#
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# logit_roc_auc = roc_auc_score(y_test, logistic_regression.predict(X_test))
# fpr, tpr, thresholds = roc_curve(y_test, logistic_regression.predict_proba(X_test)[:,1])
# plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.savefig('Log_ROC')
# plt.show()