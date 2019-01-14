
import dill as pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#filename as saved
filename = 'model_sample.pk'

#saple data for testing the model prediction
test_df = pd.read_csv('../data/test.csv', encoding="utf-8-sig")

#loading the model
with open('../models/'+filename ,'rb') as f:
    loaded_model = pickle.load(f)

#predicting and printing the results
print(loaded_model.predict(test_df))