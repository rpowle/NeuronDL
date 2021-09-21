


def save_model(model,filename):
  model_dir='model'
  os.makedirs(model_dir,exist_ok=True)
  filepath=os.path.join(model_dir,filename)
  joblib.dump(model,filepath)