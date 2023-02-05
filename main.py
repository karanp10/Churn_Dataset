import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from joblib import dump
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
import numpy as np


# Load the dataset
def clean_dataset():
    df_train = pd.read_csv('train.csv')

    hash_encoder = ce.HashingEncoder(cols=['state'])
    df_train = hash_encoder.fit_transform(df_train)
    
    df_train.international_plan.replace(['no', 'yes'], [0,1], inplace=True)
    df_train.voice_mail_plan.replace(['no', 'yes'], [0,1], inplace=True)
    df_train.churn.replace(['no', 'yes'], [0,1], inplace=True)

    onehot_area = OneHotEncoder()
    onehot_area.fit(df_train[['area_code']])

    encoded_values = onehot_area.transform(df_train[['area_code']])
    df_train[onehot_area.categories_[0]] = encoded_values.toarray()
    df_train = df_train.drop('area_code', axis=1)

    features = df_train.drop('churn', axis=1).values
    target = df_train.churn.values

    return features, target

def clean_test_dataset():
    df_test = pd.read_csv('test.csv')

    hash_encoder = ce.HashingEncoder(cols=['state'])
    df_test = hash_encoder.fit_transform(df_test)
    
    df_test.international_plan.replace(['no', 'yes'], [0,1], inplace=True)
    df_test.voice_mail_plan.replace(['no', 'yes'], [0,1], inplace=True)
    df_test.churn.replace(['no', 'yes'], [0,1], inplace=True)

    onehot_area = OneHotEncoder()
    onehot_area.fit(df_test[['area_code']])

    encoded_values = onehot_area.transform(df_test[['area_code']])
    df_test[onehot_area.categories_[0]] = encoded_values.toarray()
    df_test = df_test.drop('area_code', axis=1)

    features = df_test.drop('churn', axis=1).values
    target = df_test.churn.values

    return features, target


def load_train_split(features, target):
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.3)
    #sm = SMOTE(sampling_strategy = 1, random_state=1)

    #X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

    #scaler = MinMaxScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val

def decision_tree(X_train, X_val, y_train, y_val):
    classifier = DecisionTreeClassifier()
    classifier = classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_predictions)
    accuracy = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision').mean()

    print('DT Accuracy Score: ',accuracy)
    print('DT Precision Score: ',precision)
    print('DT Confusion Matrix: ',conf_matrix)

def random_forest(X_train, X_val, y_train, y_val):
    classifier = RandomForestClassifier(n_estimators=80, min_samples_split=5, max_depth=29)
    classifier = classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_predictions)
    accuracy = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision').mean()

    #param_dist = {
    #'n_estimators': np.arange(10, 100, 10),
    #'max_depth': np.arange(5, 30),
    #'min_samples_split': np.arange(2, 12)}

    #random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy')
    #random_search.fit(X_train, y_train)

    #best_params = random_search.best_params_
    #print(best_params)

    print('RF Accuracy Score: ',accuracy)
    print('RF Precision Score: ',precision)
    print('RF Confusion Matrix: ',conf_matrix)

def xgb(X_train, X_val, y_train, y_val):
    classifier = XGBClassifier(max_depth=9, gamma=0.480, learning_rate=0.150, min_child_weight=1, n_estimators=259, subsample=0.5851, colsample_bytree=0.541)
    classifier = classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_val)

    #param_grid = {
    #"learning_rate": uniform(0, 1),
    #"max_depth": randint(1, 10),
    #"n_estimators": randint(50, 500),
    #"min_child_weight": randint(1, 10),
    #"subsample": uniform(0.1, 1),
    #"gamma": uniform(0, 1),
    #"colsample_bytree": uniform(0.1, 1)}

    #random_search = RandomizedSearchCV(classifier, param_grid, cv=5, n_iter=100, scoring='accuracy', n_jobs=-1)

    #random_search.fit(X_train, y_train)

    #print(random_search.best_params_)

    conf_matrix = confusion_matrix(y_val, y_predictions)
    accuracy = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision').mean()

    print('XGB Accuracy Score: ',accuracy)
    print('XGB Precision Score: ',precision)
    print('XGB Confusion Matrix: ',conf_matrix)

def k_neighbors(X_train, X_val, y_train, y_val):
    classifier = KNeighborsClassifier(metric='manhattan', n_neighbors=14, weights='distance')
    classifier = classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_predictions)
    accuracy = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision').mean()

    #param_grid = {'n_neighbors': np.arange(1, 50),
    #          'weights': ['uniform', 'distance'],
    #          'metric': ['euclidean', 'manhattan']}
    
    #rand_search = RandomizedSearchCV(classifier, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1)
    #rand_search.fit(X_train, y_train)
    #best_params = rand_search.best_params_
    #best_estimator = rand_search.best_estimator_
    #best_score = rand_search.best_score_

    #print("Best parameters: ", best_params)
    #print("Best score: ", best_score)

    print('KNN Accuracy Score: ',accuracy)
    print('KNN Precision Score: ',precision)
    print('KNN Confusion Matrix: ',conf_matrix)

def ada_boost(X_train, X_val, y_train, y_val):
    classifier = AdaBoostClassifier(learning_rate=1, n_estimators=249)
    classifier = classifier.fit(X_train, y_train)
    y_predictions = classifier.predict(X_val)

    conf_matrix = confusion_matrix(y_val, y_predictions)
    accuracy = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
    precision = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision').mean()

    #param_dist = {'n_estimators': randint(50, 500),
    #          'learning_rate': [0.01, 0.1, 1, 10, 100]}
    #random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, cv=5, n_iter=10)
    #random_search.fit(X_train, y_train)

    #print("Best parameters:", random_search.best_params_)
    #print("Best score:", random_search.best_score_)


    print('ADA Accuracy Score: ',accuracy)
    print('ADA Precision Score: ',precision)
    print('ADA Confusion Matrix: ',conf_matrix)

if __name__ == "__main__":   
    features, target = clean_dataset()
    X_train, X_val, y_train, y_val = load_train_split(features, target)
    decision_tree(X_train, X_val, y_train, y_val)
    random_forest(X_train, X_val, y_train, y_val)
    xgb(X_train, X_val, y_train, y_val)
    k_neighbors(X_train, X_val, y_train, y_val)
    ada_boost(X_train, X_val, y_train, y_val)