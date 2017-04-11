#all imports, library setups here
import pandas as pd
import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
import plotly.offline as py
import plotly.graph_objs as go
from plotly.graph_objs import *

from scipy import stats, integrate
import sklearn as sk
import sklearn.cluster as cluster
from sklearn import linear_model, decomposition
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import re
import statsmodels.formula.api as sm
import seaborn as sns

sns.set(color_codes=True)
# sns.set_context('poster')
plot_kwds = {'alpha': 0.25, 's': 80, 'linewidths': 0}
regr = linear_model.LinearRegression()

print(pd.__version__)


# Read in evolution data and clean up track names
evolutionDF = pd.read_csv('EvolutionPopUSA_MainData.csv',sep=",",header='infer')
evolutionDF.track_name = [re.sub(r"\(.*\)", "", track) for track in evolutionDF.track_name]
evolutionDF.track_name = [track.strip() for track in evolutionDF.track_name]

# Read in billboard data and clean up track names
dtypespec = {"weeks": object, "peak": object, "var9": object, "var10": object, "var11": object, "var12": object, "var13": object}
billboardDF = pd.read_csv('us_billboard.csv', sep=",", header='infer', dtype=dtypespec)
# remove the garbage characters from the title field
billboardDF['track_name'] = billboardDF['track_name'].map(lambda x: str(x)[3:])
billboardDF.track_name = [re.sub(r"\(.*\)", "", track) for track in billboardDF.track_name]
billboardDF.track_name = [track.strip() for track in billboardDF.track_name]

# I only care about the tracks in the evolution dataset, but songs are 
# repeated in the billboard dataset as many times as weeks they were on 
# the list
# For each unique track name, I want the max weeks and peak, lowest 
# position, and average position
# lol how?
count = 0
uniqueTracks = evolutionDF.track_name.unique()
# temporarily take top 50 for testing
uniqueTracks = uniqueTracks[0:50]
columns = ['track_name', 'weeks', 'max_rank', 'min_rank', 'avg_rank']
uniqueTracksDF = pd.DataFrame(index=uniqueTracks, columns = columns)

for trackName in uniqueTracks:
    # df = billboardDF[billboardDF.track_name.isin([trackName])]
    # df = billboardDF[billboardDF.track_name.equals(trackName)]
    df = billboardDF[billboardDF['track_name'] == trackName]
    maxWeeks = len(df)
    maxRank = df['this_week_position'].max(axis=0)
    minRank = df['this_week_position'].min(axis=0)
    avgRank = df['this_week_position'].mean(axis=0)
    uniqueTracksDF.set_value(trackName, 'track_name', trackName)
    uniqueTracksDF.set_value(trackName, 'weeks', maxWeeks)
    uniqueTracksDF.set_value(trackName, 'max_rank', maxRank)
    uniqueTracksDF.set_value(trackName, 'min_rank', minRank)
    uniqueTracksDF.set_value(trackName, 'avg_rank', avgRank)

print(len(uniqueTracksDF))

# merge evolutionDF and uniqueTracksDF so each track in uniqueTrackDF has the corresponding evolution data
musicDF = pd.merge(evolutionDF, uniqueTracksDF, how='left', on=['track_name'])

# subset of the musicDF that only contains certain variables
# musicSubset = musicDF.ix[0,0:27].copy()
musicSubset = pd.concat([musicDF.ix[:,0:27],musicDF.ix[:,269:273]], axis=1)
# add the other features to musicSubset
musicSubset['weeks'] = musicSubset['weeks'].convert_objects(convert_numeric=True)
musicSubset['max_rank'] = musicSubset['max_rank'].convert_objects(convert_numeric=True)
musicSubset['min_rank'] = musicSubset['min_rank'].convert_objects(convert_numeric=True)
musicSubset['avg_rank'] = musicSubset['avg_rank'].convert_objects(convert_numeric=True)

# write cluster plot function to help us
musicSubset.to_csv('musicSubset.csv')
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)


clusterDF = musicSubset.ix[:, 11:27].copy()
musicSubset.to_csv('musicSubset.csv')
clusterDF.to_

clusterMatrix = clusterDF.as_matrix()


# In[269]:

#centroid based clustering
#k-means (loyds algorithm)
# assign data points to nearest cluster based on a distance metric
# calculate new centroids of clusters; repeat first step
# algorithm has converged when data points don't change cluster affiliations
# plot_clusters(clusterDF, cluster.KMeans, (), {'n_clusters': 3})
# so this doesn't seem very useful


# In[116]:

# partition into testing and training
trainingX, testingX, trainingYear, testingYear = train_test_split(clusterDF, musicSubset['year'], test_size=0.20, random_state=42)
trainingX, testingX, trainingDecade, testingDecade = train_test_split(clusterDF, musicSubset['decade'], test_size=0.20, random_state=42)
trainingX, testingX, trainingPeak, testingPeak = train_test_split(clusterDF, musicSubset['max_rank'], test_size=0.20, random_state=42)
trainingX, testingX, trainingWeeks, testingWeeks = train_test_split(clusterDF, musicSubset['weeks'], test_size=0.20, random_state=42)


regr.fit(trainingX, trainingYear)
np.mean((regr.predict(testingX)-testingYear)**2)
regr.score(trainingX, trainingYear)
regr.score(testingX, testingYear)

# hmm so 0 means no linear relationship for year
res = regr.fit(trainingX, trainingDecade)
regr.score(trainingX, trainingDecade)
regr.score(testingX, testingDecade)

res = regr.fit(trainingX, trainingWeeks)
regr.score(trainingX, trainingWeeks)
regr.score(testingX, testingWeeks)

# plt.plot(musicSubset.year.values, musicSubset.hTopic_01.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_02.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_03.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_04.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_05.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_06.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_07.values, 'go')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.hTopic_08.values, 'go')
# plt.show()
#
#
# # In[ ]:
#
# plt.plot(musicSubset.year.values, musicSubset.tTopic_01.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_02.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_03.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_04.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_05.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_06.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_07.values, 'bo')
# plt.show()
# plt.plot(musicSubset.year.values, musicSubset.tTopic_08.values, 'bo')
# plt.show()
#
#
# # In[ ]:
#
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_01.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_02.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_03.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_04.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_05.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_06.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_07.values, 'go')
# plt.show()
# plt.plot(musicSubset.max_rank.values, musicSubset.hTopic_08.values, 'go')
# plt.show()
#
#
# # In[ ]:
#
# # I'm not sure if this is relevant, but there seems to something special
# # about peaking at 100.
#
#
# # In[ ]:
#
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_01.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_02.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_03.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_04.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_05.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_06.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_07.values, 'bo')
# plt.show()
# plt.plot(musicSubset.weeks.values, musicSubset.tTopic_08.values, 'bo')
# plt.show()


# In[263]:

# musicSubset.to_csv('musicSubset.csv')


# In[10]:

sns.boxplot(x="hTopic_01", y="decade", data=musicSubset)


# In[33]:

plt.plot(musicSubset.year.values, musicSubset.weeks.values, 'bo')
plt.show()
# kind of interesting, it looks like there will be one really really long lasting song every ~5 years or so


# In[35]:

plt.plot(musicSubset.weeks.values, musicSubset.max_rank.values, 'bo')
plt.show()


# In[38]:

## MULTINOMIAL LOGISTIC REGRESSION - testing out different variable combinations
X = clusterDF
Y = musicSubset['decade']
c = 1e5

logreg = linear_model.LogisticRegression(C=c, multi_class='ovr')
logreg.fit(X, Y)
print('one-vs-rest, liblinear')
print(logreg.score(X, Y))

logreg = linear_model.LogisticRegression(C=c, multi_class='ovr', solver='newton-cg')
logreg.fit(X, Y)
print('one-vs-rest, newton-cg')
print(logreg.score(X, Y))

logreg = linear_model.LogisticRegression(C=c, multi_class='multinomial', solver='newton-cg')
logreg.fit(X, Y)
print('multinomial, newton-cg')
print(logreg.score(X, Y))

logreg = linear_model.LogisticRegression(C=c, multi_class='multinomial', solver='lbfgs')
logreg.fit(X, Y)
print('multinomial, lbfgs')
print(logreg.score(X, Y))


# USE PARTITIONED DATA
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=42)
fit_mn_logreg = linear_model.LogisticRegression(C=c, multi_class = 'multinomial', solver='newton-cg')
fit_mn_logreg.fit(Xtrain, Ytrain)
Yhat_mn = fit_mn_logreg.predict(Xtest)
print('multinomial, newton-cg')
print(fit_mn_logreg.score(Xtest,Ytest))

fit_ovr_logreg = linear_model.LogisticRegression(C=c, multi_class = 'ovr', solver='liblinear')
fit_ovr_logreg.fit(Xtrain, Ytrain)
Yhat_ovr = fit_ovr_logreg.predict(Xtest)
print('ovr, liblinear')
print(fit_ovr_logreg.score(Xtest,Ytest))



len(set(Y)) # number of decades
multinomial_conmat = confusion_matrix(Ytest, Yhat_mn)
print(multinomial_conmat)

ovr_conmat = confusion_matrix(Ytest, Yhat_ovr)
print(ovr_conmat)


# In[92]:

def analyzeConfusionMatrix(conmat):
    # Accuracy per decade
    accuracies = []
    for i in range(0, len(set(Y))):
        accuracies.append(conmat[i, i]/np.sum(conmat[i, :]))
    print("accuracy per decade")
    print(np.transpose(accuracies))

    # Accuracy/20% per decade
    accuracies = []
    for i in range(0, len(set(Y))):
        accuracies.append(conmat[i,i]/np.sum(conmat[i,:])/.2)
    print("accuracy/20% per decade")
    print(np.transpose(accuracies))

    # Precision per decade
    accuracies = []
    for i in range(0, len(set(Y))):
        accuracies.append(conmat[i,i]/np.sum(conmat[:,i]))
    print("precision per decade")
    print(np.transpose(accuracies))

analyzeConfusionMatrix(multinomial_conmat)
analyzeConfusionMatrix(ovr_conmat)


# def logRegressionPlotter(fit, X, y, multi_class):
#
#     clf = fit
#
#     # print the training scores
#     print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))
#
#     # create a mesh to plot in
#     h = .02  # step size in the mesh
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#
#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
#     plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
#     plt.axis('tight')
#
#     # Plot also the training points
#     colors = "bry"
#     for i, color in zip(clf.classes_, colors):
#         idx = np.where(y == i)
#         plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.Paired)
#
#     # Plot the three one-against-all classifiers
#     xmin, xmax = plt.xlim()
#     ymin, ymax = plt.ylim()
#     coef = clf.coef_
#     intercept = clf.intercept_
#
#     def plot_hyperplane(c, color):
#         def line(x0):
#             return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
#         plt.plot([xmin, xmax], [line(xmin), line(xmax)],
#                  ls="--", color=color)
#
#     for i, color in zip(clf.classes_, colors):
#         plot_hyperplane(i, color)
#
#     plt.show()
#
#
# # In[111]:
#
# logRegressionPlotter(fit_mn_logreg, Xtrain, Ytrain, 'newton-cg')


# In[123]:

## PCA decomosition
pca = decomposition.PCA()
pipe = sk.pipeline.Pipeline(steps=[('pca', pca), ('logistic', fit_mn_logreg)])
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')


# In[129]:

n_components = [2, 4, 8]
Cs = np.logspace(-4, 4, 3)

# Parameters of pipelines can be set using ‘__’ separated parameter names:

estimator = sk.model_selection.GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs))
estimator.fit(Xtest, Ytest)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components, linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

