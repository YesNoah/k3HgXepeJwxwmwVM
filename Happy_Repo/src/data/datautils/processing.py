from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats as stats
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
def prepro(df):
    sliced_features = df.iloc[:,1:7]
    sliced_labels = df['Y']
    sliced_features.head()

    #feature selection
    fs = SelectKBest(score_func=f_classif, k=4)
    X_selected = fs.fit_transform(sliced_features, sliced_labels)

    #Checking which features where selected
    filter = fs.get_support()
    feat = np.array(sliced_features.columns)
    print('Total Features ', feat)
    print('Selected Features for training ',feat[filter])

    #remove outliers
    z = np.abs(stats.zscore(X_selected))
    inputs = X_selected[(z<3).all(axis=1)]
    target = sliced_labels[(z<3).all(axis=1)]

    #scale inputs
    sc = StandardScaler()
    inputs = sc.fit_transform(inputs)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(inputs, target, random_state=42)

    #Balance Dataset
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)
    X_test, y_test = sm.fit_resample(X_test, y_test)
    print(X_train.shape)
    print(X_test.shape)

    #return processed dataframes
    return(X_train, X_test, y_train, y_test)

    