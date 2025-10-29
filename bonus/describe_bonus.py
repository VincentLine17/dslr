import sys
import numpy as np
from ft_statistics_bonus import ft_statistics_bonus
import pandas as pd



def ft_describe():
    
    if len(sys.argv) != 2:
        print("Please enter your dataset csv as only parameter")
        return False
    try :
        data = pd.read_csv(sys.argv[1], index_col='Index')
        # print(data.describe())
        desc = data.select_dtypes(include=['float64'])
        df = pd.DataFrame()
        columnname = list(desc.columns.values)
        for i in columnname:
            cleaned = [x for x in desc[i] if not (np.isnan(x))]
            val = []
            val.append(ft_statistics(cleaned, "count"))
            val.append(ft_statistics(cleaned, "mean"))
            val.append(ft_statistics(cleaned, "std"))
            val.append(ft_statistics(cleaned, "min"))
            val.append(ft_statistics(cleaned, "percentile", 25))
            val.append(ft_statistics(cleaned, "percentile", 50))
            val.append(ft_statistics(cleaned, "percentile", 75))
            val.append(ft_statistics(cleaned, "max"))
            dfcol = pd.DataFrame({i : val}, index =["count", "mean", "std", "min", "25%", "50%", "75%", "max"])
            if len(df) == 0:
                df = dfcol
            else:
                df = pd.concat([df, dfcol], axis = 1)
        print(df)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")



if __name__ == "__main__":
    ft_describe()