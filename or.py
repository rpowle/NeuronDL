from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot
#from utils.plot import save_plot
from utils.savee import save_model
import pandas as pd
import numpy as np


OR = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,1,1,1],
}

df = pd.DataFrame(OR)

df



X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 6

model_OR = Perceptron(eta=ETA, epochs=EPOCHS)
model_OR.fit(X, y)

_ = model_OR.total_loss()


save_model(model,filename="or.model")
save_plot (df, "or.png", model)


