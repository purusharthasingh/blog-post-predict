from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

import pandas as pd
import os
import sys

df = pd.read_csv('blogData_Train.csv', header=None)

cor = df.corr()
cor_target = abs(cor[280])
relevant_features = cor_target.sort_values(ascending=False)
n_features = 0.1*len(relevant_features)
features = relevant_features.index
features = features[1:]
print(list(features[:56]))

x = df.loc[:, :280]
y = df.loc[:,280]
x2 = StandardScaler().fit_transform(x)
print(max(x))
print(min(x))
x2 = SelectKBest(f_regression, k=28)
x3 = x2.fit(x,y)
# .fit_transform(x2, y)
print(x3.get_support(indices=True))
print(x)

res = pd.concat([x, y], axis = 1)
# matches = df.where(df.values == res.values)
# res.to_csv("top20.csv", index=False)
print (df)
# features = [str(i) for i in range(280)]


print(len(x))
print((y))

print('PCA')
n=10
pca = PCA(n_components=n)


comp = pca.fit_transform(x.iloc[:,:62])
print('Writing to file')
Pdf = pd.DataFrame(data=comp, columns = ['PCA(2) '+str(i) for i in range(1,n+1)])
res = pd.concat([Pdf, df[280]], axis = 1)
df_test = pd.read_csv('Test_data.csv', header=None)
res.to_csv("pca_28.csv", index = False, header=None)
comp = pca.fit_transform(df_test.loc[:, :62])
Pdf = pd.DataFrame(data=comp, columns = ['PCA(2) '+str(i) for i in range(1,n+1)])
res = pd.concat([Pdf, df_test[280]], axis = 1)
res.to_csv("pca_28_t.csv", index = False, header = None)
sys.exit()
print("Wrapper")

x_f = x[features[:28]]
print(x_f)
rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(x_f, y)



print('Optimal number of features: {}'.format(rfecv.n_features_))
sys.exit()

# x = StandardScaler().fit_transform(x)
# x2 = SelectKBest(chi2, k=20).fit_transform(x, y)
# print(x)

# res = pd.concat([x, y], axis = 1)
# res.to_csv("top20.csv", index=False)

