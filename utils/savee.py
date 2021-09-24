
import os
import joblib
import logging

def save_model(model,filename):
  logging.info("saving my train model")
  model_dir='model'
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  joblib.dump(model,filepath)
  logging.info(f"save the train model at : {filepath}")
  
