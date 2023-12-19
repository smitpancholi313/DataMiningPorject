#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import six
import sys
sys.modules['sklearn.externals.six'] = six
from sklearn.utils import resample
from sklearn.tree import export_graphviz
import os

# %%
# Loading the file.
startups= pd.read_csv('startups.csv', encoding='unicode_escape')

#%%
startups.info()

#%%
startupsdf = startups.rename(columns={' market ': "market", ' funding_total_usd ': "funding_total_usd"})

startupsdf['funding_total_usd']=startupsdf['funding_total_usd'].str.replace(',','') 
startupsdf['funding_total_usd']=startupsdf['funding_total_usd'].str.replace(' ','')
startupsdf['funding_total_usd']=startupsdf['funding_total_usd'].str.replace('-','0') 


startupsdf['funding_total_usd'] = pd.to_numeric(startupsdf['funding_total_usd'])


startupsdf['founded_at'] = pd.to_datetime(startupsdf['founded_at'], errors='coerce')
startupsdf['first_funding_at'] = pd.to_datetime(startupsdf['first_funding_at'], errors='coerce')
startupsdf['last_funding_at'] = pd.to_datetime(startupsdf['last_funding_at'], errors='coerce')

#%%
startupsdf['first_funding_at'].describe()
#%%
startupsdf.market = startupsdf.market.str.strip() 

#%%
startupsdf.info()
#%%
############# EXPLORATORY DATA ANALYSIS #################

startupsdf.isin([0]).sum()
startupsdf['status'].unique() 
startupsdf.groupby('status')['name'].nunique() #number of companies with each status type
#%%
# Different unique values in country_code
startupsdf['country_code'].unique() 

#%%
# Number of companies in each country_code
startupsdf.groupby('country_code')['name'].nunique().sort_values(ascending=False).head(50) 

#%%
# Grouping status and descriptive analysis of total funding

startupsdf.groupby('status')['funding_total_usd'].describe() 

#%%
# Mean values of all columns and transposing it. Grouping by company status
startupsdf.groupby('status')['funding_rounds', 'funding_total_usd', 'seed', 'venture', 'equity_crowdfunding',
       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt',
       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H'].mean().T

#%%
# Histogram of year variable
plt.figure(figsize=(10, 6))
plt.hist(startupsdf['founded_year'], bins=range(1900, 2015), color='skyblue', edgecolor='black')
plt.title('Histogram of Founded Year')
plt.xlabel('Founded Year')
plt.ylabel('Frequency')
plt.show()

#%%

############# MARKET ##############
startupsdf['market'].nunique() # 753 unique number of market
#%%
# Top 5 markets with the most funding
startupsdf.groupby('market')['funding_total_usd'].sum().sort_values(ascending = False).head(5)
#%%
# Top five markets in terms of count
startupsdf.groupby('market')['name'].count().sort_values(ascending = False).head(5) 

#%%
top_funding_markets = startupsdf.groupby('market')['funding_total_usd'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 8))
plt.pie(top_funding_markets, labels=top_funding_markets.index, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightgreen', 'lightcoral', 'lightskyblue', 'orange'])
plt.title('Distribution of Funding across Top 5 Markets')
plt.axis('equal')
plt.tight_layout()
plt.show()
#%%
# Top 5 markets in terms of count
top_count_markets = startupsdf.groupby('market')['name'].count().sort_values(ascending=False).head(5)

plt.figure(figsize=(10, 6))
top_count_markets.plot(kind='bar', color='salmon')
plt.title('Top 5 Markets with the Highest Count of Startups')
plt.xlabel('Market')
plt.ylabel('Number of Startups')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
############## REGION #############
startupsdf['region'].nunique() 

# %%
# Top 10 regions
startupsdf.groupby('region')['name'].count().sort_values(ascending = False).head(10)

# Finding the difference in days between first and last funding dates
startupsdf['diff_funding'] = startupsdf['last_funding_at'] - startupsdf['first_funding_at']
startupsdf['diff_funding'].describe() 
#%%
# turning the difference into months
startupsdf['diff_funding_months'] = (startupsdf['last_funding_at'] - startupsdf['first_funding_at'])/np.timedelta64(1, 'M') 
startupsdf['diff_funding_months'].describe()

#%%
############### TOTAL INVESTMENT ##################

startupsdf['total_investment'] = startupsdf['seed'] + startupsdf['venture'] + startupsdf['equity_crowdfunding'] + startupsdf['undisclosed'] + startupsdf['convertible_note'] + startupsdf['debt_financing'] + startupsdf['angel'] + startupsdf['grant'] + startupsdf['private_equity'] + startupsdf['post_ipo_equity'] + startupsdf['post_ipo_debt'] + startupsdf['secondary_market'] + startupsdf['product_crowdfunding']
#creating new column for total investment
startupsdf['total_investment'].describe() # calculating the total investment for each company

#%%
# sum of total investment
startupsdf['total_investment'].sum()

#%%

startupsdf['funding_total_usd'].describe()
startupsdf['funding_total_usd'].sum() # confirming that funding total and total investment is the same .

#%%
# calculating how long it took them to get their first funding after being founded
startupsdf['diff_first_funding_months'] = (startupsdf['first_funding_at'] - startupsdf['founded_at'])/np.timedelta64(1, 'M') 
startupsdf['diff_first_funding_months'].describe()

# negative values shows that there is a founded date but there is no first funding date. Median is around 17 month and mean is around 46 months.

#%%
############# COPYING DATA FRAME #################
startupsdf1 = startupsdf.copy()

startupsdf1 = startupsdf1.drop(columns= ['homepage_url', 'category_list', 'state_code', 'founded_at', 'founded_month', 'founded_quarter', 'founded_year', 
                    'diff_first_funding_months', 'diff_funding', 'funding_total_usd', 'city', 'region', 'first_funding_at', 'last_funding_at'])

startupsdf1 = startupsdf1.dropna(subset=['permalink', 'status', 'name', 'market', 'country_code', 'diff_funding_months']) 
# dropping null values from these columns

#%%
startupsdf1.isnull().sum() # There is no missing values now.
startupsdf1.shape


#%%
# making new column that has difference in funding in year
startupsdf1['diff_funding_year'] = round(startupsdf1['diff_funding_months']/12)

startupsdf1.groupby(startupsdf1['diff_funding_year'])['permalink'].count().sort_values(ascending = False).head(50)
# Only a few companies experience a funding gap exceeding 13 years.

#%%
startupsdf1.isin([0]).sum()

startupsdf2 = startupsdf1.copy() # copying the df1
startupsdf2 = startupsdf2.drop(['diff_funding_months', 'country_code', 'market'], axis=1)

startupsdf2[['funding_rounds', 'seed', 'venture', 'equity_crowdfunding',
       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',
       'private_equity', 'post_ipo_equity', 'post_ipo_debt',
       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',
       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H',
       'diff_funding_year', 'total_investment']].describe().T

#%%


################## Categorization ####################

#creating categories of these numerical values based on the output from the describe data. Also creating new column for the categories
cat_invest = pd.cut(startupsdf2.total_investment, bins = [-1, 112500, 1400300, 8205200, 40079503000], labels=['low','low_medium','high_medium','high'])
#labeling total investment values as low, low medium, high medium and high based on their descriptive summary. 
startupsdf2.insert(0,'cat_total_investment',cat_invest) # creating new column called cat_total_investment
#%%
cat_diff_funding_year = pd.cut(startupsdf2.diff_funding_year, bins = [-1, 2, 49], labels=['low','high'])
#labeling diff_funding_year as low and high based on their descriptive summary. 
startupsdf2.insert(0,'cat_diff_funding_year',cat_diff_funding_year)# creating new column called cat_diff_funding_year

#%%
cat_funding_rounds = pd.cut(startupsdf2.funding_rounds, bins = [-1, 2, 20], labels=['low','high'])
#labeling funding_rounds as low and high based on their descriptive summary. 
startupsdf2.insert(0,'cat_funding_rounds',cat_funding_rounds)# creating new column called cat_funding_rounds

#%%
cat_seed = pd.cut(startupsdf2.seed, bins = [-1, 28000, 140000000], labels=['low','high'])
#labeling seed as low and high  based on their descriptive summary. 
startupsdf2.insert(0,'cat_seed',cat_seed)# creating new column called cat_seed

#%%
cat_venture = pd.cut(startupsdf2.venture, bins = [-1, 85038.5, 6000000, 2451000000], labels=['low','medium','high'])
#labeling venture as low, medium and high based on their descriptive summary. 
startupsdf2.insert(0,'cat_venture',cat_venture) # creating new column called cat_venture

#%%
# fixing the categorical columns  into numerical values so that we can use it on the model
startupsdf2['cat_status'] = startupsdf2['status'].replace(['closed', 'operating', 'acquired'], [0, 1, 2])
startupsdf2['cat_total_investment'] = startupsdf2['cat_total_investment'].replace(['low','low_medium','high_medium','high'], [0, 1, 2, 3])
startupsdf2['cat_diff_funding_year'] = startupsdf2['cat_diff_funding_year'].replace(['low', 'high'], [0, 1])
startupsdf2['cat_funding_rounds'] = startupsdf2['cat_funding_rounds'].replace(['low', 'high'], [0, 1])
startupsdf2['cat_seed'] = startupsdf2['cat_seed'].replace(['low', 'high'], [0, 1])
startupsdf2['cat_venture'] = startupsdf2['cat_venture'].replace(['low','medium','high'], [0, 1, 3])

#%%
#as a lot of the money columns have 0, we are turning them into new categories of 0 and 1
startupsdf2.loc[startupsdf2['equity_crowdfunding'] < 1, 'cat_equity_crowdfunding'] = 0
startupsdf2.loc[startupsdf2['equity_crowdfunding'] > 1, 'cat_equity_crowdfunding'] = 1

startupsdf2.loc[startupsdf2['undisclosed'] < 1, 'cat_undisclosed'] = 0
startupsdf2.loc[startupsdf2['undisclosed'] > 1, 'cat_undisclosed'] = 1

startupsdf2.loc[startupsdf2['convertible_note'] < 1, 'cat_convertible_note'] = 0
startupsdf2.loc[startupsdf2['convertible_note'] > 1, 'cat_convertible_note'] = 1

startupsdf2.loc[startupsdf2['debt_financing'] < 1, 'cat_debt_financing'] = 0
startupsdf2.loc[startupsdf2['debt_financing'] > 1, 'cat_debt_financing'] = 1

startupsdf2.loc[startupsdf2['angel'] < 1, 'cat_angel'] = 0
startupsdf2.loc[startupsdf2['angel'] > 1, 'cat_angel'] = 1

startupsdf2.loc[startupsdf2['grant'] < 1, 'cat_grant'] = 0
startupsdf2.loc[startupsdf2['grant'] > 1, 'cat_grant'] = 1

startupsdf2.loc[startupsdf2['private_equity'] < 1, 'cat_private_equity'] = 0
startupsdf2.loc[startupsdf2['private_equity'] > 1, 'cat_private_equity'] = 1

startupsdf2.loc[startupsdf2['post_ipo_equity'] < 1, 'cat_post_ipo_equity'] = 0
startupsdf2.loc[startupsdf2['post_ipo_equity'] > 1, 'cat_post_ipo_equity'] = 1

startupsdf2.loc[startupsdf2['post_ipo_debt'] < 1, 'cat_post_ipo_debt'] = 0
startupsdf2.loc[startupsdf2['post_ipo_debt'] > 1, 'cat_post_ipo_debt'] = 1

startupsdf2.loc[startupsdf2['secondary_market'] < 1, 'cat_secondary_market'] = 0
startupsdf2.loc[startupsdf2['secondary_market'] > 1, 'cat_secondary_market'] = 1

startupsdf2.loc[startupsdf2['product_crowdfunding'] < 1, 'cat_product_crowdfunding'] = 0
startupsdf2.loc[startupsdf2['product_crowdfunding'] > 1, 'cat_product_crowdfunding'] = 1

startupsdf2.loc[startupsdf2['round_A'] < 1, 'cat_round_A'] = 0
startupsdf2.loc[startupsdf2['round_A'] > 1, 'cat_round_A'] = 1

startupsdf2.loc[startupsdf2['round_B'] < 1, 'cat_round_B'] = 0
startupsdf2.loc[startupsdf2['round_B'] > 1, 'cat_round_B'] = 1

startupsdf2.loc[startupsdf2['round_C'] < 1, 'cat_round_C'] = 0
startupsdf2.loc[startupsdf2['round_C'] > 1, 'cat_round_C'] = 1

startupsdf2.loc[startupsdf2['round_D'] < 1, 'cat_round_D'] = 0
startupsdf2.loc[startupsdf2['round_D'] > 1, 'cat_round_D'] = 1

startupsdf2.loc[startupsdf2['round_E'] < 1, 'cat_round_E'] = 0
startupsdf2.loc[startupsdf2['round_E'] > 1, 'cat_round_E'] = 1

startupsdf2.loc[startupsdf2['round_F'] < 1, 'cat_round_F'] = 0
startupsdf2.loc[startupsdf2['round_F'] > 1, 'cat_round_F'] = 1

startupsdf2.loc[startupsdf2['round_G'] < 1, 'cat_round_G'] = 0
startupsdf2.loc[startupsdf2['round_G'] > 1, 'cat_round_G'] = 1

startupsdf2.loc[startupsdf2['round_H'] < 1, 'cat_round_H'] = 0
startupsdf2.loc[startupsdf2['round_H'] > 1, 'cat_round_H'] = 1

#%%
startupsdf3 = startupsdf2[['cat_status','cat_funding_rounds',
       'cat_diff_funding_year', 'cat_total_investment' , 
       'cat_equity_crowdfunding', 'cat_venture', 'cat_seed', 'cat_undisclosed',
       'cat_convertible_note', 'cat_debt_financing', 'cat_angel', 'cat_grant',
       'cat_private_equity', 'cat_post_ipo_equity', 'cat_post_ipo_debt',
       'cat_secondary_market', 'cat_product_crowdfunding', 'cat_round_A',
       'cat_round_B', 'cat_round_C', 'cat_round_D', 'cat_round_E',
       'cat_round_F', 'cat_round_G', 'cat_round_H']] # Selecting the columns we need for the model

startupsdf3.head()

#%%
startupsdf3.isna().sum()
#%%
startupsdf4 = startupsdf3[['cat_status',
       'cat_funding_rounds', 'cat_diff_funding_year', 'cat_total_investment', 'cat_venture', 'cat_seed', 'cat_debt_financing', 'cat_angel',
       'cat_private_equity', 'cat_round_A',
       'cat_round_B', 'cat_round_C', 'cat_round_D', 'cat_round_E',
       'cat_round_F']] # selecting the columns we need
       # Created after excluding columns with less correlation

startupsdf4.shape # shape of dataset

startupsdf4.head()

#%%

#%%
startupsdf5 = startupsdf3.copy()
startupsdf5.drop(startupsdf5.index[startupsdf5['cat_status'] == 1], inplace = True)
startupsdf5 = startupsdf5.replace({'cat_status':2},1) # only 0 and 1, 0 means closed and 1 means acquired

#%%
#creating correlation matrix
colormap = plt.cm.viridis
plt.figure(figsize = (35, 35))
plt.title('Pearson Correlation of features', y = 1.05, size = 15)
matrix = np.triu(startupsdf5.corr())
sns.heatmap(startupsdf5.astype(float).corr(), linewidth = 0.1, vmax = 1.0, square =True, cmap=colormap, linecolor = 'white', annot=True, mask = matrix)

#we can remove cat_equity_crowdfunding, cat_undisclosed, cat_convertible_note, cat_grant , cat_post_ipo_equity, cat_post_ipo_debt, cat_secondary_market, cat_product_crowdfunding, cat_round_G, cat_round_H. This is the same as the other dataframe
#venture and investment is highly correlated. Also high correlation between the round and the round after it. 

#%%
startupsdf5 = startupsdf5[['cat_status', 
       'cat_funding_rounds', 'cat_diff_funding_year', 'cat_total_investment', 'cat_venture', 'cat_seed', 'cat_debt_financing', 'cat_angel',
       'cat_private_equity', 'cat_round_A',
       'cat_round_B', 'cat_round_C', 'cat_round_D', 'cat_round_E',
       'cat_round_F']] # selecting the columns we need based on the correlation matrix

#%%
Y = startupsdf4.cat_status #setting Y variable
X = startupsdf4.drop('cat_status', axis = 1) #dropping status and setting features
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)#test and train dataset

#checking size of each dataset
print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',Y_train.shape)
print('Shape of Y_test=>',Y_test.shape)

#%%
############# First Model #################
# testing with regular decision tree
clf = DecisionTreeClassifier(random_state = 100) 
clf = clf.fit(X_train, Y_train) # training decison tree classifier

preds = clf.predict(X_test) # predicting the response for test data

print(accuracy_score(Y_test,preds))
print(accuracy_score(Y_train,clf.predict(X_train)))

print('\nClassification Report\n')
print(classification_report(Y_test, preds, target_names=['Closed', 'Operating', 'Acquired']))
#accuracy score is high for training dataset which shows that it might be overfitting

#%%
#Hyper parameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 5),
    'min_samples_leaf': range(1, 5),
    'min_samples_split': range(2, 5)  
}
decision_tree = DecisionTreeClassifier()
grid = GridSearchCV(decision_tree,
                    param_grid = param_grid,
                    cv = 10, # cross validation method
                    verbose = 1,
                    n_jobs = -1) # set to use all processors

grid.fit(X_train, Y_train)

#%%
# finding the best grid parameter
grid.best_params_ 

#%%
grid.best_estimator_

#%%

########## First Model with entropy criteria ###############
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5, min_samples_leaf = 1, min_samples_split=2, random_state=40)
clf.fit(X_train,Y_train) #fitting into the model
y_train_pred=clf.predict(X_train)
y_test_pred=clf.predict(X_test)

print(f'The accuracy score of Training Accuracy is, {accuracy_score(Y_train,y_train_pred)} \nThe accuracy score of Testing Accuracy is, {accuracy_score(Y_test,y_test_pred)}') # Accuracy score of test and train data. 
print('\nClassification Report\n')
print(classification_report(Y_test, y_test_pred, target_names=['Closed', 'Operating', 'Acquired'])) # classification report


#test and train score are closer
# Training Accuracy = 86.43%
# Testing Accuracy = 86.36%

# The accuracy scores are relatively close, 
# indicating that the model performs similarly on both the training and test datasets. 
# This closeness suggests that the model is not overfitting, 
# as the performance on unseen data (test set) is comparable to the performance on the data it was trained on (training set).
#%%
############## First Model Decision Tree Visualization #######################
#visual representation of the model
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

xvar = startupsdf4.drop('cat_status', axis=1)
feature_cols = xvar.columns
plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
plot_tree(clf, filled=True, rounded=True, feature_names=feature_cols, class_names=['closed', 'operating', 'acquired'])
plt.show()

#%%
feat_importance = clf.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(feature_cols, clf.feature_importances_))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp.sort_values(by=['FeatureImportance'], ascending=False).head(20)#top 20 feature impacting decision tree split


#%%
############# Decision Tree: Binomial Classification ########################3

Y5 = startupsdf5.cat_status # need to be classified as this
X5 = startupsdf5.drop('cat_status', axis = 1) #dropping status and leaving only features
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X5, Y5, test_size = 0.2, random_state = 42)

#testing with small decision tree 
clf_pruned5 = DecisionTreeClassifier(criterion = "gini", random_state = 20,
                               max_depth=3, min_samples_leaf=5) # using depth of 3 for simplicity
clf_pruned5.fit(X_train5, Y_train5) # fitting the model
#%%
################ Data might be overfitting because of Test > Train.
preds_pruned5 = clf_pruned5.predict(X_test5)
preds_pruned_train5 = clf_pruned5.predict(X_train5)

print(accuracy_score(Y_test5, preds_pruned5)) # accuracy score of test dataset

print(accuracy_score(Y_train5, preds_pruned_train5))# accuracy score of train dataset

# accuracy score for train dataset is more than test so model might be overfitting
#%%

print('\nClassification Report\n') # Classification report
print(classification_report(Y_test5, preds_pruned5, target_names=['Class 0', 'Class 1']))
#%%
#Hyper parameter tuning

#using grid search to do hyper parameter tuning
param_grid = {
    "criterion":['gini', 'entropy'],
    "max_depth": range(1,5),
    "min_samples_split": range(2,4),
    "min_samples_leaf": range(2,4)
}

decision_tree = DecisionTreeClassifier()
grid = GridSearchCV(decision_tree,
                    param_grid = param_grid,
                    cv = 10, # cross validation method
                    verbose = 1,
                    n_jobs = -1) # set to use all processors


grid.fit(X_train5, Y_train5)

#%%
# best grid parameters
grid.best_params_ 
#%%
grid.best_estimator_
#%%

###################### Second Model ###################
#using parameter from grid to run model
clf_pruned5 = DecisionTreeClassifier(criterion = "gini", random_state = 20,
                               max_depth=3, min_samples_leaf=1, min_samples_split=2) 
clf_pruned5.fit(X_train5, Y_train5)

preds_pruned5 = clf_pruned5.predict(X_test5)
preds_pruned_train5 = clf_pruned5.predict(X_train5)
print(accuracy_score(Y_test5,preds_pruned5))#accuracy score
print(accuracy_score(Y_train5,preds_pruned_train5))# accuracy score

#classification report
print('\nClassification Report\n')
print(classification_report(Y_test5, preds_pruned5, target_names=['Class 0', 'Class 1']))

#The accurate rate came up to be 0.69 and it was good at predicting for both closed and acquired companies.
#%%
################### Decision Tree for the second model ##################
xvar5 = startupsdf5.drop('cat_status', axis=1)
feature_cols5 = xvar5.columns

plt.figure(figsize=(20, 10))  # Adjust the figure size if needed
plot_tree(clf, filled=True, rounded=True, feature_names=feature_cols, class_names=['closed', 'operating', 'acquired'])
plt.show()

# According to this model, total investment, funding rounds were important features in understanding if a company will be successful or not. 
# The model shows that total investment is very important and if it is less then the company is likely to be closed.

#%%

################### Random Forest: Multi Class Cassification ######################

########## Random Forest Model 1 ##############
Y = startupsdf4.cat_status
X = startupsdf4.drop('cat_status', axis = 1) #setting features
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)# test and train dataset

rfc = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rfc.fit(X_train, Y_train)#training the mode

rfc_pred_test = rfc.predict(X_test)
print(classification_report(Y_test, rfc_pred_test, target_names=['Closed', 'Operating', 'Acquired'])) # model is overfitting for class2 and bad at fitting 1 and 3

#%%
importances = list(rfc.feature_importances_)
feature_list = list(X.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#%%

import numpy as np

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

# Number of features to consider at every split
# Replacing 'auto' with 'sqrt', as 'auto' is not valid
max_features = ['sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

print(random_grid)


#%%
############## Random Forest: Binomial Classification #######################

Y5 = startupsdf5.cat_status# setting y variable
X5 = startupsdf5.drop('cat_status', axis = 1) # setting features
X_train5, X_test5, Y_train5, Y_test5 = train_test_split(X5, Y5, test_size = 0.2, random_state = 42) # test and train data
rfc5 = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rfc5.fit(X_train5, Y_train5) # using df5 and fitting the data
rfc_pred_test5 = rfc5.predict(X_test5) # predicting for test
print(classification_report(Y_test5, rfc_pred_test5, target_names=['Class 0', 'Class 1']))# classification report

#%%
importances = list(rfc5.feature_importances_)
feature_list = list(X5.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#Industry group, Total investment and continent name is most important features. We can only include these moving forward
#%%
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)



