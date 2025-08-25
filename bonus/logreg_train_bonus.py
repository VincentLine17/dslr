# a enlever
import matplotlib
matplotlib.use('TkAgg')

import sys
from load_csv import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

interval = 10

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

def gradient_descent(X, y, weights, iterations, learning_rate):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(weights))
        gradient = (1/m) * X.T.dot(h - y)
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

def stochastic_gradient_descent(X, y, weights, iterations=1000, learning_rate=0.01, interval=100):
    m = len(y)
    cost_history = []
    prev_cost = float("inf")

    for epoch in range(iterations):
        i = np.random.randint(m)
        xi = X[i:i+1]
        yi = y[i:i+1]
        h = sigmoid(xi.dot(weights))
        gradient = xi.T.dot(h - yi)
        weights -= learning_rate * gradient.flatten()
        if epoch % interval == 0:
            cost = compute_cost(X, y, weights)
            cost_history.append(cost)
            if abs(prev_cost - cost) < 1e-6:
                print(f"Arrêt anticipé à l’epoch {epoch}")
                break
            prev_cost = cost
    return weights, cost_history

def mini_batch_gradient_descent(X, y, weights, iterations, learning_rate, batch_size=32):
    m = len(y)
    cost_history = []
    for iter in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            h = sigmoid(X_batch.dot(weights))
            gradient = (1 / len(y_batch)) * X_batch.T.dot(h - y_batch)
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
        weights_all_gd = np.zeros((X.shape[1], 4))
        weights_all_sgd = np.zeros((X.shape[1], 4))
        weights_all_minib = np.zeros((X.shape[1], 4))

        cost_histories = {
            "Gradient Descent": [],
            "Stochastic Gradient Descent": [],
            "Mini-Batch": []
        }
        for i in range(4):
            y_i = Y[:, i]
            weights_gd, cost_gd = gradient_descent(X, y_i, weights_initial.copy(), 300, 0.05)
            weights_all_gd[:, i] = weights_gd
            cost_histories["Gradient Descent"].append(cost_gd)
            weights_sgd, cost_sgd = stochastic_gradient_descent(X, y_i, weights_initial.copy(), 300 * len(y_i), 0.05)
            weights_all_sgd[:, i] = weights_sgd
            cost_histories["Stochastic Gradient Descent"].append(cost_sgd)
            weights_minib, cost_minib = mini_batch_gradient_descent(X, y_i, weights_initial.copy(), 300, 0.05, 32)
            weights_all_minib[:, i] = weights_minib
            cost_histories["Mini-Batch"].append(cost_minib)

        for method, histories in cost_histories.items():
            plt.figure(figsize=(10, 6))
            plt.plot(histories[0], label=f"{method} - Gryffindor", color='red')
            plt.plot(histories[1], label=f"{method} - Hufflepuff", color='yellow')
            plt.plot(histories[2], label=f"{method} - Ravenclaw", color='blue')
            plt.plot(histories[3], label=f"{method} - Slytherin", color='green')
            plt.xlabel("Epochs")
            plt.ylabel("Coût")
            plt.title("Comparaison Batch vs Stochastic Gradient Descent vs Mini-Batch")
            plt.legend()
            plt.grid(True)
            plt.show()

        np.savetxt('weights_gd.txt', weights_all_gd)
        np.savetxt('weights_sto.txt', weights_all_sto)
        np.savetxt('weights_minb.txt', weights_all_minb)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()