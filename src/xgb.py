from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Integer encode columns with 2 unique values

import pandas as pd

df = pd.read_csv("../data/heart_2020_cleaned.csv")  ## 使用相对路径读取数据。“..”代表上一层路径


encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                      '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                      '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                      '30-34':32,'25-29':27}
df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
df['AgeCategory'] = df['AgeCategory'].astype('float')



for col in ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']:
    if df[col].dtype == 'O': # 是类别特征。int float 是数值特征
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
# One-hot encode columns with more than 2 unique values
df = pd.get_dummies(df, columns=['Race', 'Diabetic', 'GenHealth', ], prefix = ['Race', 'Diabetic', 'GenHealth'])


scaler = MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()


from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, RocCurveDisplay
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score



y = scaled_df['HeartDisease']
del scaled_df['HeartDisease']
X = scaled_df
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score,recall_score,roc_auc_score

def plot_roc(y_test, y_test_pred,label):
    auc = roc_auc_score(y_test, y_test_pred) # calculate the auc roc score
    print("AUC: ", auc)
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, y_test_pred, pos_label=1) #function to create the roc curve

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))] #random variable
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1) #plot actual values vs random predictions
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--',color='orange') #plot false positive rate vs true positive rate
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC',dpi=300)
    plt.show()

if __name__ == "__main__":
    models = [KNeighborsClassifier(), LogisticRegression(), XGBClassifier(),GradientBoostingClassifier(),SVC(),AdaBoostClassifier(),DecisionTreeClassifier()]
    # scores = dict()
    #
    # svc = SVC(probability=True)
    # svc.fit(X_train, y_train)
    # y_pred = svc.predict_proba(X_test)
    # print(f'model: {str(m)}')
    # for model in models:
    #     model.fit(X_train, y_train)
    #     y_pred_proba = [i[-1] for i in model.predict_proba(X_test)]
    #     y_pred = model.predict(X_test)
    #
    #     print(y_pred)
    #     print(f'model: knn classfication')
    #     print(f'Accuracy_score: {accuracy_score(y_test, y_pred)}')
    #     print(f'Precission_score: {precision_score(y_test, y_pred)}')
    #     print(f'Recall_score: {recall_score(y_test, y_pred)}')
    #     print(f'F1-score: {f1_score(y_test, y_pred)}')
    #     print("roc", roc_auc_score(y_test, y_pred_proba))
    #     print('-' * 30, '\n')
    #     plot_roc(y_test, y_pred_proba,label)
    # Create LogisticRegression
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    lr_y_predict = model.predict(X_test)

    print(f'model: ' + '' )
    print(f'Accuracy_score: {accuracy_score(y_test, lr_y_predict)}')
    print(f'Precission_score: {precision_score(y_test, lr_y_predict)}')
    print(f'Recall_score: {recall_score(y_test, lr_y_predict)}')
    print(f'F1-score: {f1_score(y_test, lr_y_predict)}')
    ax = plt.gca()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    # rdf_disp = RocCurveDisplay.from_estimator(clf, x_test, y_test, ax=ax)
    # lg_disp = RocCurveDisplay.from_estimator(lr, x_test, y_test, ax=ax)
    plt.show()
