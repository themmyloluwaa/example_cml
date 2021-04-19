
# In[ ]:


# In[ ]:


import logging
from joblib import dump
import datetime
import shap
import autosklearn.classification
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly
plotly.__version__


# In[ ]:


# In[ ]:


data_path = "/data/"


# In[ ]:


model_path = "models/"


# In[ ]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[ ]:


logging.basicConfig(
    filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# Please Download the data from [this source](https://drive.google.com/file/d/1af2YyHIp__OdpuUeOZFwmwOvCsS0Arla/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Loading Data and Train-Test Split](#P0)
#

# In[ ]:


df = pd.read_csv(f"{data_path}data-breast-cancer.csv")


# In[ ]:


# encode the categrical column
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])


# In[ ]:


df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)


# In[ ]:


test_size = 0.2
random_state = 45


# In[ ]:


train, test = train_test_split(
    df, test_size=test_size, random_state=random_state)


# In[ ]:


logging.info(
    f'train test split with test_size={test_size} and random state={random_state}')


# In[ ]:


train.to_csv(f'{data_path}Breast_Cancer_Train.csv', index=False)


# In[ ]:


train = train.copy()


# In[ ]:


test.to_csv(f'{data_path}Breast_Cancer_Test.csv', index=False)


# In[ ]:


test = test.copy()


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[ ]:


X_train, y_train = train.iloc[:, 1:], train['diagnosis']


# In[ ]:


total_time = 600
per_run_time_limit = 30


# In[ ]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[ ]:


logging.info(
    f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved classification model at {model_path}model{timesstr}.pkl ')


# In[ ]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


# In[ ]:


X_test, y_test = train.iloc[:, 1:], train['diagnosis']


# Now, we can attempt to predict the diagnosis prediction from our test set. To do that, we just use the .predict method on the object "automl" that we created and trained in the last sections:

# In[ ]:


y_pred = automl.predict(X_test)


# Let's now evaluate it using the mean_squared_error function from scikit learn:

# In[ ]:

with open("metrics.txt", 'w') as outfile:
    outfile.write(
        f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")

logging.info(
    f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# we can also plot the y_test vs y_pred scatter:

# In[ ]:


df = pd.DataFrame(np.concatenate(
    (X_test, y_test.to_numpy().reshape(-1, 1), y_pred.reshape(-1, 1)),  axis=1))


# In[ ]:


df.columns = ['radius_mean', 'texture_mean', 'perimeter_mean',
              'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
              'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
              'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
              'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
              'fractal_dimension_se', 'radius_worst', 'texture_worst',
              'perimeter_worst', 'area_worst', 'smoothness_worst',
              'compactness_worst', 'concavity_worst', 'concave points_worst',
              'symmetry_worst', 'fractal_dimension_worst', 'Predicted Target', 'True Target']


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(
    f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(
    model=automl.predict, data=X_test.iloc[:50, :], link="identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(
    X=X_test.iloc[X_idx:X_idx+1, :], nsamples=100)
X_test.iloc[X_idx:X_idx+1, :]
# print the JS visualization code to the notebook
# shap.initjs()
shap.force_plot(base_value=explainer.expected_value,
                shap_values=shap_value_single,
                features=X_test.iloc[X_idx:X_idx+1, :],
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(
    f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X=X_test.iloc[0:50, :], nsamples=100)


# In[ ]:


# print the JS visualization code to the notebook
# shap.initjs()
fig = shap.summary_plot(shap_values=shap_values,
                        features=X_test.iloc[0:50, :],
                        show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(
    f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")


# --------------
# # End of This Notebook
