import seaborn
import matplotlib.pyplot as plt
import pandas as pd
from load_csv_bonus import load

try:
    data = load("dataset_train.csv")
    # data = load("visu_houses.csv")

    data = data.drop(columns=["Arithmancy"])
    data = data.drop(columns=["Astronomy"])
    # data = data.drop(columns=["Herbology"])
    # data = data.drop(columns=["Defense Against the Dark Arts"])
    data = data.drop(columns=["Divination"])
    data = data.drop(columns=["Muggle Studies"])
    # data = data.drop(columns=["Ancient Runes"])
    data = data.drop(columns=["History of Magic"])
    data = data.drop(columns=["Transfiguration"])
    data = data.drop(columns=["Potions"])
    data = data.drop(columns=["Care of Magical Creatures"])
    # data = data.drop(columns=["Charms"])
    # data = data.drop(columns=["Flying"])

    column = data.select_dtypes(include=['float64'])
    columnname = list(column.columns.values)

    df = data["Hogwarts House"]
    for i in columnname:
        df = pd.concat([df, data[i]], axis = 1)

    seaborn.pairplot(df, hue ='Hogwarts House', hue_order=["Ravenclaw", "Hufflepuff", "Slytherin", "Gryffindor"], diag_kind="hist")
    plt.show()

except Exception as e:
    print(f"{type(e).__name__}: {e}")