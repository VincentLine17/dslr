import matplotlib
matplotlib.use('TkAgg')

import sys
from load_csv import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_rate = 0.5
iterations = 1000

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

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))
    cost = -(1/m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
    return cost

def gradient_descent(X, y, weights):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(weights))
        gradient = (1/m) * X.T.dot(h - y)
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

def main():
    
    if len(sys.argv) != 2:
        print("Please enter your dataset csv as only parameter")
        return False
    try :
        liste = load(sys.argv[1])
        if liste.shape[1] < 2:
            print(f"Your {sys.argv[1]} file hasn't enough data")
            return
        print(f"Enter a feature for train between these: {liste.columns.tolist()}")
        while True:
            feature1 = sys.stdin.readline().rstrip('\n')
            if feature1 in liste.columns:
                print("Enter a second feature")
                while True:
                    feature2 = sys.stdin.readline().rstrip('\n')
                    if feature1 == feature2:
                        print("Enter a diferent feature from first one")
                        continue
                    if feature2 == 'Hogwarts House':
                        print("Hogwarts House not accepted")
                        continue
                    if feature2 in liste.columns:
                        break
                break

        df = liste[feature1]
        df = pd.concat([df, liste[feature2]], axis = 1)
        df = pd.concat([df, liste['Hogwarts House']], axis = 1)

        df = df.dropna()
        print(df.shape)

        X = df[[feature1, feature2]].values
        X = normalize_min_max(X)
        X = np.c_[np.ones(X.shape[0]), X]
        
        house_int = df['Hogwarts House'].map(house_to_int).values

        Y = np.zeros((house_int.size, 4))
        Y[np.arange(house_int.size), house_int] = 1
        
        weights_initial = np.random.randn(X.shape[1]) * 0.01

        weights_all = np.zeros((X.shape[1], 4))
        for i in range(4):
            y_i = Y[:, i]
            weights, cost_history = gradient_descent(X, y_i, weights_initial)
            weights_all[:, i] = weights      

        np.savetxt('weights.txt', weights_all)

        plt.plot(range(iterations), cost_history, label="Fonction de coût")
        plt.xlabel('Itérations')
        plt.ylabel('Coût')
        plt.title('Evolution de la fonction de coût pendant l\'entraînement')
        plt.grid(True)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()