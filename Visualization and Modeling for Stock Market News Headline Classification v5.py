#!/usr/bin/env python
# coding: utf-8

# In[143]:


# Added comment for git testing
#Natural Language Processing with the DIJA and Reddit Headlines
#Classification Predictions on Stock Market from Headlines
#Classification includes Overall Up or Down, Market Volitality, and Measure of Strong and Poor Days 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns 

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#cross validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

#accuracy
from sklearn.metrics import accuracy_score

#train a perceptron model 
from sklearn.linear_model import Perceptron

#plot decision regions to visualize
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS
import re
import nltk
from nltk.corpus import stopwords

#estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

#model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[436]:


#Import Data for Viz
df = pd.read_csv('Combined_News_DJIA.csv')
CATdf = pd.read_csv('Corrected_Combined_DJIA_cat.csv')
corrCATdf = pd.read_csv('corrCATdf.csv')

#Import Data for Modeling
data = pd.read_csv('Combined_News_DJIA.csv')
dataCAT = pd.read_csv('Corrected_Combined_DJIA_cat.csv')
DJIAdf = pd.read_csv('DJIA_table.csv')
corrCATdf = pd.read_csv('corrCATdf.csv')


# In[145]:


CATdf.info()


# In[4]:


#Define features and Dependent Variable
CATdf_features = CATdf.iloc[:,2:35]
depVar = CATdf['NetUpDown']


# In[5]:


#Define X, Y using features and dependent variable
X = CATdf_features
y = depVar


# In[6]:


#Number of unique y values 
print('Class labels:', np.unique(y))


# In[7]:


# Create a default pairplot
sns.pairplot(DJIAdf)


# In[8]:


# Take the log of Volume and Close 
DJIAdf['log_vol'] = np.log10(DJIAdf['Volume'])
DJIAdf['log_close'] = np.log10(DJIAdf['Close'])
sns.pairplot(DJIAdf)

# Drop the non-transformed columns
DJIAdf = DJIAdf.drop(columns = ['Volume'])
DJIAdf = DJIAdf.drop(columns = ['Adj Close'])
DJIAdf = DJIAdf.drop(columns = ['Close'])
DJIAdf = DJIAdf.drop(columns = ['Open'])
DJIAdf = DJIAdf.drop(columns = ['High'])
DJIAdf = DJIAdf.drop(columns = ['Low'])


# In[9]:


# Create a pairplot with Volume/Close
sns.pairplot(DJIAdf)


# In[10]:


#Correlation heat map
#Visulaize highly correlated features (quantitative realtion between two features)

corrCATdf_table = CATdf.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corrCATdf_table, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(corrCATdf_table.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corrCATdf_table.columns)
ax.set_yticklabels(corrCATdf_table.columns)
plt.show()


# In[11]:


#Correlation Table
#View numeric values of correlation 
print(corrCATdf_table)


# In[13]:


#Covariance Heat Map 
#visualize coavriance (measure of how two features change together)

covCATdf_table = CATdf.cov()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(covCATdf_table,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(covCATdf_table.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(covCATdf_table.columns)
ax.set_yticklabels(covCATdf_table.columns)
plt.show()


# In[14]:


#Covariance Table
#View numeric values of Covariance 
covCATdf_table = CATdf.cov()
print(covCATdf_table)


# In[16]:


#Factor Plot of NetUpDown
sns.catplot('NetUpDown', data = CATdf, kind = 'count')


# In[17]:


#Count Net totals
NetUpDown_totals = CATdf.groupby('NetUpDown')['NetUpDown'].count()
NetUpDown_totals
#More Positive Days than Negative days


# In[18]:


#Factor Plot of HLcat
sns.catplot('HLcat', data = CATdf, kind = 'count')


# In[19]:


#Count Net totals
HLcat_totals = CATdf.groupby('HLcat')['HLcat'].count()
HLcat_totals
#1 = 0-100 in points swing
#2 = 100 - 250
#3 = 250+


# In[20]:


#Another way to plot histogram of HLdifference
CATdf['HLdifference'].hist(bins = 100)
#Shows numbers of days of HLdifference


# In[434]:


#Guassian curve manufactured for HLdifference 

# histogram plot of a low res sample
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
from numpy import exp
from scipy.stats import boxcox
# seed the random number generator
seed(1)

#define data
data2 = CATdf.HLdifference
#power transform
data1 = boxcox(data2, 0)

pyplot.hist(data1)
pyplot.show()


# In[21]:


#Factor Plot of OCcat
sns.catplot('OCcat', data = CATdf, kind = 'count')


# In[22]:


#Count totals of OCcat
OCcat_totals = CATdf.groupby('OCcat')['OCcat'].count()
OCcat_totals
#2 = >100 (very positive day)
#1 = >0 (positive day)
#-1 = >-100 (negative day)
#-2 = <-100 (very negative day)


# In[23]:


#Another way to plot histogram of OCdifference
CATdf['OCdifference'].hist(bins = 100)
#Shows numbers of days of OCdifference


# In[25]:


#Distribution of age, with an overlay of a density plot
volume = CATdf['Volume'].dropna()
volume_dist = sns.distplot(volume)
volume_dist.set_title("Distribution of Trade Volume")
#Bell curve of numbers of days with Volume


# In[26]:


#Guassian curve manufactured for Volume 
# histogram plot of a low res sample
from numpy.random import seed
from numpy.random import randn
from matplotlib import pyplot
from numpy import exp
from scipy.stats import boxcox
# seed the random number generator
seed(1)

#define data
data2 = CATdf.Volume
#power transform
data1 = boxcox(data2, 0)

pyplot.hist(data1)
pyplot.show()


# In[27]:


#Plot Trade HLDifference Over Time (volatility)
import datetime

X = pd.to_datetime(CATdf.Date)
y = CATdf.HLdifference

#plot
plt.plot(X,y)


# In[28]:


#Plot Trade OCdifference Over Time
import datetime

X = pd.to_datetime(CATdf.Date)
y = CATdf.OCdifference

#plot
plt.plot(X,y)
plt.gcf().autofmt_xdate()
plt.show()


# In[29]:


#Show Close over Time
import datetime

X = pd.to_datetime(CATdf.Date)
y = CATdf.Close

#plot
plt.plot(X,y)
plt.gcf().autofmt_xdate()
plt.show()


# In[32]:


#Linear Plot of Volume and HLcat on Market Up/Down
#illustrates low volatilty days more liekly to finish net positive
#also higher volume on low and high volatility days more likely to finish net positive

sns.lmplot('Volume', 'NetUpDown', data=CATdf, hue = 'HLcat')


# In[33]:


#Graph DJIA Close with HLdiffernce and Volume for insight

index = pd.read_csv('djia_df_cat.csv')

index.Date = pd.to_datetime(index.Date)
plt.figure(figsize=(10,8))
plt.plot(index.Date, index.Close,label = "DJIA closing price");
plt.plot(index.Date, index.HLdifference*10,label = "HLDifference"); #scale volume for readability
plt.plot(index.Date, index.Volume/100000, label = "Volume");
plt.legend();
plt.title("DJIA stocks");


# In[34]:


#BEGIN MODELING

#split data set train/test
train = dataCAT[dataCAT['Date'] < '2015-01-01']
test = dataCAT[dataCAT['Date'] > '2014-12-31']


# In[35]:


train.describe()


# In[36]:


test.describe()


# In[43]:


#Process of breaking down headlines into CountVector array below 
example = train.iloc[0,17]
print(example)


# In[44]:


#Make all lowercase
example2 = example.lower()
print(example2)


# In[45]:


#Split words using CountVectorizer
example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)


# In[46]:


#Remove Stop Words
example4 = [word for word in example3 if word not in stopwords.words('english')] 


# In[47]:


print(example4)


# In[435]:


#Illustration of One-Hot Encoding used to create an array (example with stop words)
vectorEX = CountVectorizer()
EX = vectorEX.fit_transform(example3)
print(EX.toarray())


# In[48]:


#Illustration of One-Hot Encoding used to create an array and later help remove stop words (example without stop words)
vectorEX = CountVectorizer()
EX = vectorEX.fit_transform(example4)
print(EX.toarray())


# In[49]:


#Islotaed Words and Count the Number of times they appear 
pd.DataFrame([[x,example4.count(x)] for x in set(example4)], columns = ['Word', 'Count'])


# In[437]:


#Calculate the values 0.25 quantile and 0.75 quantile of Stock Market Data
DJIAdf.quantile([0.25, 0.75])


# In[219]:


#Define Different Algorithms Below for later Modeling

#Random Forest 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=300, criterion='gini', 
                            max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                            bootstrap=True, oob_score=True, n_jobs=1, random_state=1, 
                            verbose=0, warm_start=False, class_weight=None)


# In[220]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', 
                           algorithm='auto', leaf_size=30, p=2, 
                           metric='minkowski', metric_params=None, n_jobs=None)


# In[221]:


#Multi-layer Perceptron classifier

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=300, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.001, validation_fraction=0.1, verbose=False,
       warm_start=False)


# In[222]:


#C-Support Vector Classification

import sklearn.svm as svm

sv = svm.LinearSVC(penalty='l2', loss='squared_hinge', 
                    dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
                    fit_intercept=True, intercept_scaling=1, 
                    class_weight=None, verbose=0, random_state=1, max_iter=1000)


# In[56]:


#Use for loop to iterate through each row of the dataset
#combine all headlines into a single string

trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,11:35]))


# In[58]:


#add that string to the list we need for CountVectorizer
onewordvector = CountVectorizer()
onewordtrain = onewordvector.fit_transform(trainheadlines)
print(onewordtrain.shape)

#show array of number of rows and total number of unique words in trainheadlines


# In[60]:


#Train a logistic Regression model
#name the model then fit the model based on X and Y values
#Sub LogisticRegression() with different defined algo for comapring results between algos


onewordmodel = LogisticRegression()
onewordmodel = onewordmodel.fit(onewordtrain, train["NetUpDown"]) #Also change y-value here for comparing other 
                                                                  #classification categories


# In[61]:


#repeat steps used to prep training data 
#predict whether the DJIA increased or decreased for each day in test dataset
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,11:35]))


# In[62]:


#add that string to the list we need for CountVectorizer
onewordtest = onewordvector.transform(testheadlines)
predictions = onewordmodel.predict(onewordtest)


# In[63]:


print(onewordtest.shape)


# In[64]:


#Look at predictions using crosstab
pd.crosstab(test["NetUpDown"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[65]:


#Show accuracy 
#Be sure to label correct y-value being tested

acc1 = accuracy_score(test['NetUpDown'], predictions)
print('One Word Model Accuracy: ', acc1)


# In[68]:


#Identify the Top 10 Positive and Negative coefficients
#Bag of Words
#For LogisticRegression() only

onewordwords = onewordvector.get_feature_names()
onewordcoeffs = onewordmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : onewordwords,
                        'Coefficient' : onewordcoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending = [0, 1])
#positive words
coeffdf.head(10)


# In[69]:


#Negative words
coeffdf.tail(10)


# In[438]:


#Two-Word Modeling, using words paired together
#n-gram model, n = length of sequence of words to be counted
#n = 2 model
twowordvector = CountVectorizer(ngram_range = (2,2))
twowordtrain = twowordvector.fit_transform(trainheadlines)


# In[439]:


#view data 
print(twowordtrain.shape)
#Shows an new array with two-word combinations (now 355,342)


# In[73]:


#Name and fit Model  Two Word Model 
#Sub LogisticRegression() with different defined algo for comapring results between algos

twowordmodel = LogisticRegression()
twowordmodel = twowordmodel.fit(twowordtrain, train["NetUpDown"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
twowordtest = twowordvector.transform(testheadlines)
twowordpredictions = twowordmodel.predict(twowordtest)


# In[74]:


#Cross tab results
pd.crosstab(test["NetUpDown"], twowordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[75]:


#Show accuracy 
acc2 = accuracy_score(test['NetUpDown'], twowordpredictions)
print('Two Word Model accuracy: ', acc2)


# In[76]:


#Word Pairing coefficients
twowordwords = twowordvector.get_feature_names()
twowordcoeffs = twowordmodel.coef_.tolist()[0]
twowordcoeffdf = pd.DataFrame({'Words' : twowordwords,
                          'Coefficient' : twowordcoeffs})
twowordcoeffdf = twowordcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
#Positive Word Pairings
twowordcoeffdf.head(10)


# In[77]:


#Negative Word Pairings
twowordcoeffdf.tail(10)


# In[78]:


#Three Word Modeling
#n-gram model, n = length of sequence of words to be counted
#n = 3 model
threewordvector = CountVectorizer(ngram_range = (3,3))
threewordtrain = threewordvector.fit_transform(trainheadlines)


# In[80]:


#view data 
print(threewordtrain.shape)
#589,589 unique three-word combinations


# In[81]:


#Name and fit Model Three Word Model 
threewordmodel = LogisticRegression()
threewordmodel = threewordmodel.fit(threewordtrain, train["NetUpDown"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
threewordtest = threewordvector.transform(testheadlines)
threewordpredictions = threewordmodel.predict(threewordtest)


# In[82]:


#Cross tab results
pd.crosstab(test["NetUpDown"], threewordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[83]:


#Show accuracy 
acc3 = accuracy_score(test['NetUpDown'], threewordpredictions)
print('Three Word Model accuracy: ', acc3)


# In[84]:


#Three Word coefficients

threewordwords = threewordvector.get_feature_names()
threewordcoeffs = threewordmodel.coef_.tolist()[0]
threewordcoeffdf = pd.DataFrame({'Words' : threewordwords,
                          'Coefficient' : threewordcoeffs})
threewordcoeffdf = threewordcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
#Positive Words
threewordcoeffdf.head(10)


# In[85]:


#Negative words
threewordcoeffdf.tail(10)


# In[86]:


#Model for OCcat (Open/Close category: -2, -1, 1, 2)
#Showing Most accurate model 

#Two-Word Modeling, using words paired together
#n-gram model, n = length of sequence of words to be counted
#n = 2 model

twowordvector = CountVectorizer(ngram_range = (2,2))
twowordtrain = twowordvector.fit_transform(trainheadlines)


# In[87]:


#view data 
print(twowordtrain.shape)


# In[88]:


#Name and fit Model 
#Using Support Vector on new y-variable OCcat
twowordmodel = sv
twowordmodel = twowordmodel.fit(twowordtrain, train["OCcat"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
twowordtest = twowordvector.transform(testheadlines)
twowordpredictions = twowordmodel.predict(twowordtest)


# In[89]:


#Cross tab results
pd.crosstab(test["OCcat"], twowordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[90]:


#Show accuracy for OCcat Prediction
acc2 = accuracy_score(test['OCcat'], twowordpredictions)
print('Two Word Model accuracy on OCcat: ', acc2)


# In[440]:


#Model for HLcat (High/Low category measuring volatility 1, 2, 3)
#Showing most accuracte HLcat model 

onewordvector = CountVectorizer()
onewordtrain = onewordvector.fit_transform(trainheadlines)
print(onewordtrain.shape)

#shows total number of different words (31,122)


# In[441]:


#Train one word model on HLcat using Random Forest (Best Performing Algo)
onewordmodel = rf
onewordmodel = onewordmodel.fit(onewordtrain, train["HLcat"])


# In[442]:


#repeat steps used to prep training data 
#predict whether the DJIA increased or decreased for each day in test dataset
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,11:35]))


# In[443]:


#add that string to the list we need for CountVectorizer
onewordtest = onewordvector.transform(testheadlines)
predictions = onewordmodel.predict(onewordtest)


# In[444]:


#Look at predictions using crosstab
pd.crosstab(test["HLcat"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[445]:


#Show accuracy 
#Be sure to label correct y-value being tested

acc1 = accuracy_score(test['HLcat'], predictions)
print('One Word Model Accuracy HLcat: ', acc1)


# In[446]:


#Show Two Word Model for HLcat using SVM 
#Showing second most accurate HLcat model 

twowordvector = CountVectorizer(ngram_range = (2,2))
twowordtrain = twowordvector.fit_transform(trainheadlines)


# In[450]:


#Two word Model SVM on new y-variable HLcat
twowordmodel = sv
twowordmodel = twowordmodel.fit(twowordtrain, train["HLcat"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
twowordtest = twowordvector.transform(testheadlines)
twowordpredictions = twowordmodel.predict(twowordtest)


# In[451]:


#Cross tab results
pd.crosstab(test["HLcat"], twowordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[452]:


#Show accuracy for HLcat Prediction
acc2 = accuracy_score(test['HLcat'], twowordpredictions)
print('Two Word Model accuracy on HLcat: ', acc2)


# In[ ]:


##MODELING WITHOUT STOP WORDS


# In[453]:


#Total number of stop words 
print(len(stopwords.words('english')))


# In[454]:


#All the stop words in 'english'
print((stopwords.words('english')))


# In[101]:


#Remove Stop Words from trainheadlines

def stopremovedheadlines(trainheadlines1):
    
    trainheadlines1 = [CountVectorizer(lowercase = True).build_tokenizer()(line) for line in trainheadlines]

    trainheadlines2 = []
    for line in trainheadlines1:
        temp = []
        for word in line:
            temp.append(word.lower())
        trainheadlines2.append(temp)

    nostopwords = []

    counter = 0
    for line in trainheadlines2:
        #if counter % 100 == 0: print(counter)
        temp = []
        for word in line:
            if word not in stopwords.words('english'):
                temp.append(word)
        new = ' '.join(temp)
        nostopwords.append(new)
        counter += 1
        
    return nostopwords


# In[148]:


#Define trainheadlines with no stop words
trainheadlinesNOSTOP = stopremovedheadlines(trainheadlines)


# In[455]:


#Confirm correct length for train set 1611
print(len(trainheadlinesNOSTOP))


# In[456]:


#add that string to the list we need for CountVectorizer
onewordvector = CountVectorizer()
onewordtrain = onewordvector.fit_transform(trainheadlinesNOSTOP)

#confirm numbers of rows 1611
#140, the number of stop words removed from original 31,122
#30982
print(onewordtrain.shape)


# In[457]:


#repeat steps used to prep training data 
#predict whether the DJIA increased or decreased for each day in test dataset
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,11:35]))


# In[458]:


#Remove Stop Words from testheadlines
def stopremovedheadlines1(testheadlines):
    
    testheadlines1 = [CountVectorizer(lowercase = True).build_tokenizer()(line) for line in testheadlines]

    testheadlines2 = []
    for line in testheadlines1:
        temp = []
        for word in line:
            temp.append(word.lower())
        testheadlines2.append(temp)

    nostopwords = []

    counter = 0
    for line in testheadlines2:
        #if counter % 100 == 0: print(counter)
        temp = []
        for word in line:
            if word not in stopwords.words('english'):
                temp.append(word)
        new = ' '.join(temp)
        nostopwords.append(new)
        counter += 1
        
    return nostopwords


# In[158]:


#Define testheadlines with no stop words
testheadlinesNOSTOP = stopremovedheadlines1(testheadlines)


# In[159]:


print(len(testheadlinesNOSTOP))


# In[465]:


#One Word No Stop Words Model 
#Sub LogisticRegression() with different defined algo for comapring results between algos


onewordmodel = LogisticRegression()
onewordmodel = onewordmodel.fit(onewordtrain, train["NetUpDown"]) #Also change y-value here for comparing other 
                                                                  #classification categories


# In[466]:


#add that string to the list we need for CountVectorizer
onewordtest = onewordvector.transform(testheadlinesNOSTOP)
predictions = onewordmodel.predict(onewordtest)


# In[467]:


print(onewordtest.shape)
#shows 30,982 unique words


# In[468]:


#Look at predictions using crosstab
pd.crosstab(test["NetUpDown"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[470]:


#Show accuracy 
#Be sure to label correct y-value being tested

acc1 = accuracy_score(test['NetUpDown'], predictions)
print('One Word Model Accuracy: ', acc1)


# In[471]:


#Identify the Top 10 Positive and Negative coefficients
#Bag of Words
#For LogisticRegression() only

onewordwords = onewordvector.get_feature_names()
onewordcoeffs = onewordmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : onewordwords,
                        'Coefficient' : onewordcoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending = [0, 1])
#positive word
coeffdf.head(10)


# In[472]:


#Negative word
coeffdf.tail(10)


# In[473]:


#Two-Word Modeling no Stop Words
#n-gram model, n = length of sequence of words to be counted
#n = 2 model

twowordvector = CountVectorizer(ngram_range = (2,2))
twowordtrain = twowordvector.fit_transform(trainheadlinesNOSTOP)


# In[474]:


#view data 
print(twowordtrain.shape)
#354,664 unique two-word combinations


# In[475]:


#Name and fit Model 
#Sub LogisticRegression() with different defined algo for comapring results between algos

twowordmodel = LogisticRegression()
twowordmodel = twowordmodel.fit(twowordtrain, train["NetUpDown"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
twowordtest = twowordvector.transform(testheadlinesNOSTOP)
twowordpredictions = twowordmodel.predict(twowordtest)


# In[476]:


#Cross tab results
pd.crosstab(test["NetUpDown"], twowordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[477]:


#Show accuracy 
acc2 = accuracy_score(test['NetUpDown'], twowordpredictions)
print('Two Word Model accuracy: ', acc2)


# In[478]:


#Word Pairing coefficients

twowordwords = twowordvector.get_feature_names()
twowordcoeffs = twowordmodel.coef_.tolist()[0]
twowordcoeffdf = pd.DataFrame({'Words' : twowordwords,
                          'Coefficient' : twowordcoeffs})
twowordcoeffdf = twowordcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
#Positive Word Pairings
twowordcoeffdf.head(10)


# In[479]:


#Negative Word Pairings
twowordcoeffdf.tail(10)


# In[381]:


#Three Word Modeling No Stop Words
#n-gram model, n = length of sequence of words to be counted
#n = 3 model

threewordvector = CountVectorizer(ngram_range = (3,3))
threewordtrain = threewordvector.fit_transform(trainheadlinesNOSTOP)


# In[382]:


#view data 
print(threewordtrain.shape)

#441,541 unique variables representing three-word combinations


# In[480]:


#Name and fit Model 
threewordmodel = LogisticRegression()
threewordmodel = threewordmodel.fit(threewordtrain, train["NetUpDown"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
threewordtest = threewordvector.transform(testheadlinesNOSTOP)
threewordpredictions = threewordmodel.predict(threewordtest)


# In[481]:


#Cross tab results
pd.crosstab(test["NetUpDown"], threewordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[482]:


#Show accuracy 
acc3 = accuracy_score(test['NetUpDown'], threewordpredictions)
print('Three Word Model accuracy: ', acc3)


# In[483]:


#Word Pairing coefficients

threewordwords = threewordvector.get_feature_names()
threewordcoeffs = threewordmodel.coef_.tolist()[0]
threewordcoeffdf = pd.DataFrame({'Words' : threewordwords,
                          'Coefficient' : threewordcoeffs})
threewordcoeffdf = threewordcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
#Positive Word Pairings
threewordcoeffdf.head(15)


# In[484]:


#Negative word pairings
threewordcoeffdf.tail(10)


# In[500]:


#One Word No Stop Words Model 
#Showing Most accurate HLcat model
#Using Multi-Layer Perceptron Model 
#Sub LogisticRegression() with different defined algo for comapring results between algos


onewordmodel = mlp
onewordmodel = onewordmodel.fit(onewordtrain, train["HLcat"]) #Also change y-value here for comparing other 
                                                                  #classification categories


# In[501]:


#add that string to the list we need for CountVectorizer
onewordtest = onewordvector.transform(testheadlinesNOSTOP)
predictions = onewordmodel.predict(onewordtest)


# In[502]:


print(onewordtest.shape)
#shows 30,982 unique words


# In[503]:


#Look at predictions using crosstab
pd.crosstab(test["HLcat"], predictions, rownames=["Actual"], colnames=["Predicted"])


# In[504]:


#Show accuracy 
#Be sure to label correct y-value being tested

acc1 = accuracy_score(test['HLcat'], predictions)
print('One Word Model Accuracy: ', acc1)


# In[505]:


#Two Word Model no Stop Words
#Shows most accurate OCcat model 
#Using Multi-Layer Perceptron Algo 
#Sub LogisticRegression() with different defined algo for comapring results between algos

twowordmodel = mlp
twowordmodel = twowordmodel.fit(twowordtrain, train["OCcat"])

#transfrom test data
testheadlines = []
for row in range(0, len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row, 11:35]))
twowordtest = twowordvector.transform(testheadlinesNOSTOP)
twowordpredictions = twowordmodel.predict(twowordtest)


# In[506]:


#Cross tab results
pd.crosstab(test["OCcat"], twowordpredictions, rownames = ["Actual"], colnames=["Predicted"])


# In[507]:


#Show accuracy 
acc2 = accuracy_score(test['OCcat'], twowordpredictions)
print('Two Word Model accuracy: ', acc2)

