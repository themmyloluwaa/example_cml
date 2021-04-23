# # -*- coding: utf-8 -*-
# """SIT_W2D1_Breast_Cancer_Solution.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1WXsdndBENoHeUHkE7z4LkL8ia9odVqrd

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Breast Cancer Selection</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>

# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice binary classification on Breast Cancer data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>

# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/17X_OTM8Zqg-r4XEakCxwU6VN1OsJpHh7?usp=sharing" title="momentum"> Assignment, Classification of breast cancer cells</a>
# </strong></nav>

# ## Submitted by Temiloluwa Ojo and Robiya Farmonova

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)
# """

# !sudo apt-get install build-essential swig
# !curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
# !pip install auto-sklearn
# !pip install -U matplotlib
# !pip install pipelineprofiler
# !pip install shap
# !pip install --upgrade plotly
# !pip3 install -U scikit-learn

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, KFold
# from pandas_profiling import ProfileReport
# import matplotlib.pyplot as plt
# import plotly
# plotly.__version__

# import plotly.graph_objects as go
# import plotly.io as pio
# import plotly.express as px
# from plotly.subplots import make_subplots

# # your code here
# from scipy import stats
# from sklearn.preprocessing import  StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn import preprocessing
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# import time
# from google.colab import files

# from sklearn import set_config
# from sklearn.compose import ColumnTransformer

# import autosklearn.classification
# import PipelineProfiler
# import shap

# """**Connect** to your Google Drive"""

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

# data_path = "/content/drive/MyDrive/Introduction2DataScience/data/"

# pd.set_option('display.max_rows', 20)

# set_config(display='diagram')

# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline

# """Please Download the data from [this source](https://drive.google.com/file/d/1af2YyHIp__OdpuUeOZFwmwOvCsS0Arla/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Exploratory Data Analysis](#P0)

# ### Understand the Context

# **What type of problem are we trying to solve?**

# With this data set, we want to build a classifier that would predict if a given sample is from a malignant or benign tumor.


# **_This is a binary classification problem_**

# **How was the data collected?/ Is there documentation on the Data?**

# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
# n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

# This database is also available through the UW CS ftp server:
# ftp ftp.cs.wisc.edu
# cd math-prog/cpo-dataset/machine-learn/WDBC/

# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

# Attribute Information:

# 1. ID number
# 2. Diagnosis (M = malignant, B = benign)

# **3-32**

# _Ten real-valued features are computed for each cell nucleus:_

# 1. radius (mean of distances from center to points on the perimeter)
# 2. texture (standard deviation of gray-scale values)
# 3. perimeter
# 4. area
# 5. smoothness (local variation in radius lengths)
# 6. compactness (perimeter^2 / area - 1.0)
# 7. concavity (severity of concave portions of the contour)
# 8. concave points (number of concave portions of the contour)
# 9. symmetry
# 10. fractal dimension ("coastline approximation" - 1)

# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.

# All feature values are recoded with four significant digits.

# Missing attribute values: none

# Class distribution: 357 benign, 212 malignant

# **Do we have assumption about the data?**

# * Type of columns - we may have few categorical features e.g. diagnosis - benign or malignat but as we read most of them seem to be numeric data

# * 357+212=569  we may have not much data to train

# * We may have high correlation amon some features. e.g. among radius, perimeter and area

# **Can we foresee any challenge related to this data set?**

# * As mentioned above, we may have high correlation between some features and it can create challenges analysing this data

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**
# """

# df = pd.read_csv(f"{data_path}data-breast-cancer.csv")

# """**How many columns and rows do we have?**"""

# df.shape

# """**What are the names and meaning of each columns?**"""

# df.columns

# """print the first 10 rows of the dataframe"""

# df.head(10)

# """**What are the types of each column?**"""

# # your code here
# df.info()

# """- As we expected, except the column "diagnosis", all columns' type is numerical, specifically "float64".
# - The type of the column "diagnosis" is object, specifically string.
# - Almost all columns are non-null except the last column "Unnamed 32" which is good to achieve a good predictions.
# - We can drop the last column because it doesn't give us any inofrmation with null values for all rows

# **Do the types correspond to what you expected?
# if not, which columns would you change and why?**

# - Yes almost the types correspond to what we expected except the column "Unnamed 32" which doesn't give us any information. So as the first change we can drop this column.
# - As the second change we should transform the "diagnosis" column to numerical values e.g. benighn - 0, malignant - 1 because our machine learning algorithms work with numerical data.
# - Also we drop the first column ID which won't give us any neccessary information to train properly our dataset.

# **Perform the necessary type transformations**

# Nothing here

# **What are the possible categories for categorical columns?/What is the min, max and mean of each numerical columns?**
# """

# df.describe(include='all',datetime_is_numeric=True)

# """_Your Comments here_
# * From what we can observe, the column "diagnosis" possible catagories will be malignant - 1 and benign - 0 that would predict if a given sample is from a malignant or benign tumor.

# * Also we can see some outliers here. e.g. where the radius_worst = 36.04 the area_worst = 4254 but mathematically is impossible why because area_worst = pi*r^2 = 3,14*36.04^2 ~ 4078

# let's just have a look at the categories for all columns of type object:
# """

# for column in df.select_dtypes(include=['object']):
#     print(f'Column {column} has {len(df[column].unique())} categories: {df[column].unique()}\n')

# """Let's create a list of all categorical variables, for later use:"""

# categories = ['diagnosis']

# """**Perform test/train split here**

# !!! Please think about it!!! How should the data be splitted?

# Observing the target column:
# """

# df['diagnosis'].value_counts()

# """It would be concerning when we split our data because the target column contains data with a big difference which could cause one split having more benign data which could lead to overfiitting or bias. We can ensure this doesn't happen using Scikitlearn stratify method.

# <!-- _Your Comments here_

# We will also drop the column "Unnamed 32"  with all NaN values which gives us nothing and "id". Then seperated the columns: y - 'diagnosis' to classify whether benign or malignant. Then we splitted the data as we exlplained above to the train and test sets - 20% of the data to test the model and 80% of the data to train our model.  -->
# """

# # your code here

# #dropping some columns and separating them
# df.drop(['Unnamed: 32','id'], axis=1, inplace=True)
# y = df['diagnosis']
# X = df.drop(['diagnosis'], axis=1)

# """ Observing the size of X and Y variable"""

# # observe this well
# print("Size of dataset: ", df.shape)
# print("Size of X: ", X.shape)
# print("Size of y: ", y.shape)

# """now, we can perform the split, stratifying according to the target (y) data:"""

# X_train, X_test, y_train, y_test = train_test_split(X, # original dataframe to be split
#                                                      y,
#                                                      test_size=0.2, # proportion of the rows to put in the test set
#                                                      stratify=y,
#                                                      random_state=45) # for reproducibility (see explanation below)

# """from now on, we will use the train dataframe. let's save the train and test datasets in the data folder:"""

# X_train.to_csv(f'{data_path}Cancer_X_Train.csv', index=False)

# y_train.to_csv(f'{data_path}Cancer_y_Train.csv', index=False)

# X_test.to_csv(f'{data_path}Cancer_X_Test.csv', index=False)

# y_test.to_csv(f'{data_path}Cancer_y_Test.csv', index=False)

# """### Missing Values and Duplicates

# **Are there some duplicate columns? rows?**
# """

# # your code here
# #to find duplicated columns
# df.columns.duplicated()

# #to find duplicated rows
# duplicateDFRow = df[df.duplicated()]
# print(duplicateDFRow)

# """As we can see we don't have any duplicated columns and rows in our dataframe so in our train set too.

# We can also confirm this by printing the columns in X_train, and first 10 rows in X_train
# """

# # Columns of X_train
# print("Columns of X_train: \n\n", X_train.columns)

# # First 10 rows of X_train
# X_train.head(10)

# """**Should we drop duplicate rows?**

# There are no duplicate rows, so there's no need to drop any

# **How many missing values are there in each columns?**
# """

# print(df.isna().sum())

# """we can also use the count method, which counts the number of rows that have a value (the non-missing data!): """

# print(X_train.count().to_string())

# """As we can see in our dataset we don't any missing data in each column.

# ### Data Distribution and Outliers

# **What is the distribution of numerical/categorical data?**

# Let's plot histograms of all the features with the proportion of diagnosis of Malignant and Benign patients:
# """

# for column in X_train:
#     fig = px.histogram(X_train, x=column, color=y_train)
#     fig.update_layout(width=700, height=300)
#     fig.show()

# fig = px.histogram(y_train, x='diagnosis')
# fig.update_layout(width=700, height=300)
# fig.show()

# """We can clearly see variations of the ratio of malignant/benign for several variables. That's really promising for breast cancer prediction!

# **Are there clear outliers?**
# """

# columns_1 = ['compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'smoothness_mean']
# plt.figure(figsize=(20,10))
# df.boxplot(columns_1)

# columns_2 = ['radius_mean', 'texture_mean', 'perimeter_mean']
# plt.figure(figsize=(20,10))
# df.boxplot(columns_2)

# column3 = ['area_mean']
# df.boxplot(column3)

# """Yes, we can see clear outliers in our dataset e.g. in the example of area mean - area_mean >= 2500

# **Can we rule out some outliers as mistakes in the data collecting process?**

# _Your answer here_

# **How should we deal with outliers?**

# We replace the outliers with Mean values of each column with mean values of the columns.

# ### Relationship between features (correlations)

# **What are the relationships between features (make a pairplot)? Are they linear?**
# """

# import seaborn as sns

# sns.pairplot(data=df, hue='diagnosis', kind='scatter')
# # to show
# plt.show()

# X_train.corr().style.background_gradient(cmap='coolwarm')

# """As we can see the relationship of the some features are linear perimeter & area and radius & perimeter and area&radius but not all of them

# **What correlation coefficients should be computed?**

# Standard Pearson Correlation Coefficients are fine!

# **Is there risk of data leakage?**

# None

# ### Feature Creation and Combination

# - **What kind of Scaling should we use/try?**
# - **Should we transform some features?**
# - **Should we drop some features?**
# - **Should we combine features?**

# When building Machine learning models, we need to encode categorical variables into numerical values that can be fed to a model.
# Since our variables have no order, we can use one hot encoding.
# """

# # # your code here
# # #encode string columns

# # encoder = LabelEncoder()
# # df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

# """ <!-- _Your Comments here_

# We transformed only the columnn of "diagnosis" to numerical data. 0 - benign, 1- malignant. All other colums are not categorical columns. Regarding last "Unnamed 32 " column we didn't why because we will simply drop this column wothout any transformations. -->
# """

# X_train.columns

# # Print features with catagory type
# for column in X_train.select_dtypes(include=['object']):
#     print(f'Column {column} has {len(X_train[column].unique())} categories: {X_train[column].unique()}\n')

# """From the above output, the train datasets has no categorical feature that needs encoding or transformation.

# **_What do you observe?_**

# From the above output, the train datasets has no categorical feature that needs encoding or transformation.

# Transformation would be done one only the y variable.
# """

# num_variables = [ 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
#        'smoothness_mean', 'compactness_mean', 'concavity_mean',
#        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
#        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
#        'fractal_dimension_se', 'radius_worst', 'texture_worst',
#        'perimeter_worst', 'area_worst', 'smoothness_worst',
#        'compactness_worst', 'concavity_worst', 'concave points_worst',
#        'symmetry_worst', 'fractal_dimension_worst']

# # ohe_variables = ['diagnosis']

# """### Conclusion: Experimental setup and  Possible Feature Transformations

# Let's wrap up on the exploratory data analysis and conclude. We should now be able to answer the following questions:

# - **What would be our baseline for the analysis?**
# - **What kind of modelling setup should we use/try?**
# - **What kind of Scaling should we use/try?**
# - **If outliers, what kind of treatment should we apply?**
# - **Should we transform some features?**
# - **Should we drop some features?**
# - **Should we combine features?**

# **_write a small paragraph answering these questions_**

# Our problem is a binary classification problem. We could start by using the logistic regression model.

# We ordered the relevant variables into 2 groups:

# - ohe_variables: variables to encode with the OneHotEncoder
# - num_variables: numerical variables.

# No column in our dataset contains any missing value

# Finally, we will have to scale our numerical variables, so that each column we consider in our model has the same weight.

# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# ### Pipeline Definition
# """

# numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
#                                       ('scaler', StandardScaler())])

# # ohe_transformer = OneHotEncoder(handle_unknown='ignore')

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, num_variables)
#         ])

# classification_model = Pipeline(steps=[('preprocessor', preprocessor),
#                                           ('classifier', LogisticRegression())])

# classification_model

# """#### Model Cross Validation"""

# cross_val_score(classification_model, X_train, y_train)

# """### Model Training"""

# # Your code here
# classification_model.fit(X_train, y_train)

# """#### autoML"""

# # your code here
# columns = num_variables.copy()

# X_train_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_train), columns=columns)

# """Encode feature 'diagnosis' with label encoder"""

# le = preprocessing.LabelEncoder()
# le.fit(y)
# y_train_encoded = le.transform(y_train)

# automl = autosklearn.classification.AutoSklearnClassifier(
#     time_left_for_this_task=600,
#     per_run_time_limit=30,
# )

# automl.fit(X_train_encoded, y_train_encoded)

# profiler_data= PipelineProfiler.import_autosklearn(automl)
# PipelineProfiler.plot_pipeline_matrix(profiler_data)

# """### Model Evaluation"""

# X_test_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_test), columns=columns)

# y_pred = automl.predict(X_test_encoded)

# y_test_encoded = le.transform(y_test)

# confusion_matrix(y_test_encoded,y_pred)

# ConfusionMatrixDisplay(confusion_matrix(y_test_encoded,y_pred))

# explainer = shap.KernelExplainer(model = automl.predict, data = X_test_encoded.iloc[:50, :], link = "identity")

# # Set the index of the specific example to explain
# X_idx = 0
# shap_value_single = explainer.shap_values(X = X_test_encoded.iloc[X_idx:X_idx+1,:], nsamples = 100)
# X_test.iloc[X_idx:X_idx+1,:]
# # print the JS visualization code to the notebook
# shap.initjs()
# shap.force_plot(base_value = explainer.expected_value,
#                 shap_values = shap_value_single,
#                 features = X_test_encoded.iloc[X_idx:X_idx+1,:]
#                 )

# shap_values = explainer.shap_values(X = X_test_encoded.iloc[0:50,:], nsamples = 100)

# # print the JS visualization code to the notebook
# shap.initjs()
# shap.summary_plot(shap_values = shap_values,
#                   features = X_test_encoded.iloc[0:50,:]
#                   )

# """_Your Comments here_

# --------------
# # End of This Notebook
# """
