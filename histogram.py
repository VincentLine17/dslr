import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from load_csv import load


try:
    liste = load('dataset_train.csv')
    Gryffindor = liste[liste['Hogwarts House'] == "Gryffindor"]
    Slytherin = liste[liste['Hogwarts House'] == "Slytherin"]
    Ravenclaw = liste[liste['Hogwarts House'] == "Ravenclaw"]
    Hufflepuff = liste[liste['Hogwarts House'] == "Hufflepuff"]

    Gryffindor = Gryffindor.select_dtypes(include=['float64'])
    Slytherin = Slytherin.select_dtypes(include=['float64'])
    Ravenclaw = Ravenclaw.select_dtypes(include=['float64'])
    Hufflepuff = Hufflepuff.select_dtypes(include=['float64'])

    data = liste.select_dtypes(include=['float64'])
    columnname = list(data.columns.values)

    i = "Care of Magical Creatures"
    plt.hist([Gryffindor[i], Slytherin[i], Ravenclaw[i], Hufflepuff[i]], histtype='stepfilled', stacked=True)
    plt.xlabel(f"{i} distribution")
    plt.ylabel("Frequency")
    plt.show()

    # for i in columnname:
    #     plt.hist([Gryffindor[i], Slytherin[i], Ravenclaw[i], Hufflepuff[i]], histtype='stepfilled', stacked=True)
    #     plt.xlabel(f"{i} distribution")
    #     plt.ylabel("Frequency")
    #     plt.show()

except Exception as e:
    print(f"{type(e).__name__}: {e}")