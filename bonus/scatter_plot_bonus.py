import matplotlib.pyplot as plt
import pandas as pd
from load_csv_bonus import load

try:
    # data = load("dataset_train.csv")
    liste = load("visu_houses.csv")
    Gryffindor = liste[liste['Hogwarts House'] == "Gryffindor"]
    Slytherin = liste[liste['Hogwarts House'] == "Slytherin"]
    Ravenclaw = liste[liste['Hogwarts House'] == "Ravenclaw"]
    Hufflepuff = liste[liste['Hogwarts House'] == "Hufflepuff"]

    Gryffindor = Gryffindor.select_dtypes(include=['float64'])
    Slytherin = Slytherin.select_dtypes(include=['float64'])
    Ravenclaw = Ravenclaw.select_dtypes(include=['float64'])
    Hufflepuff = Hufflepuff.select_dtypes(include=['float64'])

    print(liste)
    liste = liste.set_index("Hogwarts House", drop=True)
    print(liste)
    data = liste.select_dtypes(include=['float64'])
    columnname = list(data.columns.values)

    plt.scatter(Gryffindor['Astronomy'], Gryffindor['Defense Against the Dark Arts'], alpha=0.5)
    plt.scatter(Slytherin['Astronomy'], Slytherin['Defense Against the Dark Arts'], alpha=0.5)
    plt.scatter(Ravenclaw['Astronomy'], Ravenclaw['Defense Against the Dark Arts'], alpha=0.5)
    plt.scatter(Hufflepuff['Astronomy'], Hufflepuff['Defense Against the Dark Arts'], alpha=0.5)
    plt.xlabel(f"{'Astronomy'}")
    plt.ylabel(f"{'Defense Against the Dark Arts'}")
    plt.show()

except Exception as e:
    print(f"{type(e).__name__}: {e}")
