import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy 
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression

dataset_cols=["bike_count","hour","temp","humidity","wind","visibility","dew_pt_temp","radiation","rain","snow","functional"]
df = pd.read_csv("SeoulBikeData.csv").drop(["Date","Holiday","Seasons"],axis=1)

df.columns = dataset_cols
df["functional"] = (df["functional"]=="Yes").astype(int)
df = df[df["hour"]==12]
df = df.drop(["hour"], axis=1)
df.head()

for label in df.columns[1:]:
    plt.scatter(df[label],df["bike_count"])
    plt.title(label)
    plt.ylabel("Bike count at n0oon")
    plt.xlabel("label")
    plt.legend()
    plt.show()


df=df.drop(["wind","visibility","functional"], axis=1)
df.head()

train, val, test = np.split(df.sample(frac=1),[int(0.6*len(df)),int(0.8*len(df))])

def get_xy(dataframe,y_label,x_labels=None):
    dataframe= copy.deepcopy(dataframe)
    if not x_labels:
        x= dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels)==1:
            x= dataframe[x_labels[0]].values.reshape(-1,1)
        else:
            x= dataframe[x_labels].values

    y= dataframe[y_label].values.reshape(-1,1)
    data= np.hstack((x,y))

    return data, x, y

_, x_train_temp , y_train_temp = get_xy(train, "bike_count",x_labels=["temp"])
_, x_val_temp , y_val_temp = get_xy(val,"bike_count",x_labels=["temp"])
_, x_test_temp, y_test_temp= get_xy(test,"bike_count",x_labels=["temp"])

