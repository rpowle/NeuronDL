import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import joblib # for saving method
plt.style.use("fivethirtyeight")# this is style of graphs
import os

def prepare_data(df):
  X = df.drop("y", axis=1)

  y = df["y"]

  return X, y


  def save_model(model,filename):
  model_dir='model'
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  joblib.dump(model,filepath)



  def save_model(model,filename):
  model_dir='model'
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  joblib.dump(model,filepath)

