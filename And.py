from utils.model import Perceptron
from utils.all_utils import prepare_data,save_plot
#from utils.plot import save_plot
from utils.savee import save_model
import pandas as pd
import numpy as np






def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(AND)
    print(df)
    X,y = prepare_data(df)
    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)
    save_model(model,filename="And.model")
    save_plot(df, "and.png", model)




if __name__ == '__main__':
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y": [0,0,0,1],
}
ETA = 0.3 # 0 and 1
EPOCHS = 10

main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)