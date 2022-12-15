import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import time
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

RANDOM_STATE = 12
FOLDS = 5
TARGET = 'Transported'

data = pd.concat([train, test], axis=0, ignore_index=True)

exp_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

data['Surname'] = data['Name'].str.split().str[-1]
data.loc[data['Surname'] == 'Unknown', 'Surname'] = np.nan
data = pd.concat([data, data['Cabin'].str.split('/', expand=True)], axis=1)
data.rename(columns={0: 'Deck', 1: 'Num', 2: 'Side'}, inplace=True)
data['Expenditure'] = data[exp_feats].sum(axis=1)
data['No_spending'] = (data['Expenditure'] == 0).astype(int)

# Preprocessing NaN values


# Filling NaN in HomePlanet column
data['Group'] = data['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
GHP_gb = data.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
GHP_index = data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index
data.loc[GHP_index, 'HomePlanet'] = data.iloc[GHP_index, :]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

data.loc[(data['HomePlanet'].isna()) & (data['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
data.loc[(data['HomePlanet'].isna()) & (data['Deck'] == 'G'), 'HomePlanet'] = 'Earth'

SHP_gb = data.groupby(['Surname', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)
SHP_index = data[data['HomePlanet'].isna()][(data[data['HomePlanet'].isna()]['Surname']).isin(SHP_gb.index)].index
data.loc[SHP_index, 'HomePlanet'] = data.iloc[SHP_index, :]['Surname'].map(lambda x: SHP_gb.idxmax(axis=1)[x])

HPD_gb = data.groupby(['HomePlanet', 'Destination'])['Destination'].size().unstack().fillna(0)
data.loc[(data['HomePlanet'].isna()) & ~(data['Deck'] == 'D'), 'HomePlanet'] = 'Earth'
data.loc[(data['HomePlanet'].isna()) & (data['Deck'] == 'D'), 'HomePlanet'] = 'Mars'

# Filling NaN in Destinations column

data.loc[(data['Destination'].isna()), 'Destination'] = 'TRAPPIST-1e'

# Filling NaN in VIP column

data.loc[data['VIP'].isna(), 'VIP'] = False

# Filling NaN in CryoSleep

na_rows_CSL = data.loc[data['CryoSleep'].isna(), 'CryoSleep'].index
data.loc[data['CryoSleep'].isna(), 'CryoSleep'] = \
    data.groupby(['No_spending'])['CryoSleep'].transform(lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CSL]

# Impute continuous data with NaN value by MEDIAN

imputer = SimpleImputer(strategy='median')
imputer_cols = ["Age", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "RoomService"]
imputer.fit(data[imputer_cols])
data[imputer_cols] = imputer.transform(data[imputer_cols])

# Encoding Categorical Features

label_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]


def label_encoder(info, columns):
    for col in columns:
        info[col] = info[col].astype(str)
        info[col] = LabelEncoder().fit_transform(info[col])
    return info


data = label_encoder(data, label_cols)

# Spilt data

data.drop(["Name", "Cabin", "PassengerId", "Surname", "Num", "Expenditure", "No_spending", "Group", "Deck", "Side"],
          axis=1, inplace=True)
train = data.iloc[:train.shape[0], :]
test = data.iloc[train.shape[0]:, :]
X = train.drop(columns='Transported')
y = train['Transported'].astype(int)
pred = test.drop(columns='Transported')
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30)


rf_clf = RandomForestClassifier()
rf_clf.fit(x_train, y_train)
print("RandomForest Classifier validation accuracy: ", accuracy_score(y_valid, rf_clf.predict(x_valid).astype(bool)))

cb_clf = CatBoostClassifier()
cb_clf.fit(x_train, y_train)
print("Catboost Classifier validation accuracy: ", accuracy_score(y_valid, rf_clf.predict(x_valid).astype(bool)))

y_pred = cb_clf.predict(pred).astype(bool)
submission['Transported'] = y_pred
submission.to_csv("submission.csv", index=False)
