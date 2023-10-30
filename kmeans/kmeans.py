# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split 

# %%
dataset_dir = '../PySpark/ETL_result'
dir_contents = os.listdir(dataset_dir)
dir_contents.sort()
for f in dir_contents:
	print(f) if f.endswith('.csv') else None

# %%
twitter_col = ['account_id', 'followers_count', 'following_count', \
    'post_count', 'listed_count', 'active_date']
df = pd.DataFrame(columns = twitter_col)

for f in dir_contents:
    if f.endswith('.csv'):
        df1 = pd.read_csv(dataset_dir + '/' + f, names = twitter_col)
        df = pd.concat([df, df1])

# %%
df.isnull().sum()

# %%
df.dropna(subset = ['account_id'], inplace=True)
df['active_date'].fillna(df['active_date'].mean(), inplace=True)
df.info()

# %%
df['post_count'] = df['post_count'].astype(float)
df['listed_count'] = df['listed_count'].astype(float)
print(df.info())
print(df.head())

# %%
features = df.iloc[:,1:]
print(features.info())
print(features.head())

# %%
features_norm = MinMaxScaler().fit_transform(features)
features_norm = pd.DataFrame(features_norm, columns = list(features.columns))
print('Normalized Features')
print(features_norm.head())

# %%
# use PCA to convert dimension to 2
pca = PCA(2)
data = pca.fit_transform(features_norm)

# fitting multiple k-means algorithms
k = 15
model = KMeans(n_clusters = k, init='k-means++')
model.fit(data)
pred = model.predict(data)
uniq = np.unique(pred)

centers = np.array(model.cluster_centers_)

# plot clusters
plt.figure(figsize=(12,12))

for i in uniq:
   plt.annotate(i, (centers[:,0][i], centers[:,1][i]), fontsize=18)
   plt.scatter(data[pred == i , 0], data[pred == i , 1], s = 2, label = i)
plt.scatter(centers[:,0], centers[:,1], marker='x', color='k')
plt.legend()
plt.show()

# %%
frame = features.copy()
frame['cluster'] = pred
print('Cluster counts')
print(frame['cluster'].value_counts())

# %%
center = frame.groupby(['cluster']).mean()
center['cluster'] = uniq
center = center.sort_values(['followers_count','listed_count','post_count'], ascending=True)
# add ranking
center['rank'] = uniq + 1
# export json
center.to_json('center_location.json', orient='records', lines=True)

center

# %%
ranking = pd.DataFrame(columns=['cluster','rank'])
ranking['cluster'] = center['cluster']
ranking['rank'] = center['rank']
ranking = np.array(ranking)
ranking

# %%
frame['rank'] = 0
for i in range(0,len(ranking)):
    frame['rank'].loc[frame['cluster'] == ranking[i][0]] = ranking[i][1]

# %%
frame.head(30)

# %%
frame.to_csv('frame_with_ranking.csv')

# %%



