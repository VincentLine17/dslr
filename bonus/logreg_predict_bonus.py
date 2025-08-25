import matplotlib
matplotlib.use('TkAgg')

import sys
from load_csv import load
import numpy as np
import pandas as pd

house_to_int = {
    'Gryffindor': 0,
    'Hufflepuff': 1,
    'Ravenclaw': 2,
    'Slytherin': 3
}

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def normalize_min_max(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def predict(X, weights):
    scores = X.dot(weights)
    probabilites = sigmoid(scores)
    return np.argmax(probabilites, axis=1)

def main():

    if len(sys.argv) != 3:
        print("Please enter your dataset csv to test and the file containing weights previously trained")
        return False
    try :
        weights_all = np.loadtxt(sys.argv[2], delimiter=' ')
        if weights_all.shape != (3, 4):
            print("Weight file should be of shape [3, 4]")
            return False
        if np.isnan(weights_all).any():
            print("NaN values not accepted")
            return False
        listepred = load(sys.argv[1])
        listepred = listepred.select_dtypes(include=['float64'])
        if listepred.shape[1] < 2:
            print(f"Your {sys.argv[1]} file hasn't enough data")
            return
        print(f"Enter a feature for prediction between these: {listepred.columns.tolist()}")
        while True:
            feature1 = sys.stdin.readline().rstrip('\n')
            if feature1 in listepred.columns:
                print("Enter a second feature for prediction")
                while True:
                    feature2 = sys.stdin.readline().rstrip('\n')
                    if feature1 == feature2:
                        print("Enter a diferent feature from first one")
                        continue
                    if feature2 in listepred.columns:
                        break
                break

        dfpred = listepred[feature1]
        dfpred = pd.concat([dfpred, listepred[feature2]], axis = 1)
        dfpred = dfpred.dropna()

        Xpred = dfpred[[feature1, feature2]].values
        Xpred = normalize_min_max(Xpred)
        Xpred = np.c_[np.ones(Xpred.shape[0]), Xpred]

        predictions = predict(Xpred, weights_all)

        listepred = listepred.dropna(subset=[feature1, feature2])
        
        int_to_house = {v: k for k, v in house_to_int.items()}
        
        house_names = np.vectorize(int_to_house.get)(predictions)
        listepred['Hogwarts House'] = house_names
        listepred.to_csv('visu_houses.csv')
        
        houses = listepred.iloc[:, :1]
        houses.to_csv('houses.csv')
    
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()