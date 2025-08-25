from math import sqrt, ceil
import numpy as np

def ft_count(num) -> float:
    i = 0
    for n in num:
        i += 1
    return i

def ft_mean(num) -> float:
    tot = 0
    i = 0
    for n in num:
        tot += float(n)
        i += 1
    if i == 0:
        print(f"ERROR")
        return float('nan')
    else :
        return tot / i
        print(f"mean : {tot / i}")

def ft_median(num) -> float:
    num = sorted(num)
    if len(num) == 0:
        print(f"ERROR")
        return float('nan')
    else :
        return num[int(len(num) / 2)]
        print(f"median : {num[int(len(num) / 2)]}")

def ft_min(num) -> float:
    if len(num) == 0:
        return float('nan')
    min = num[0]
    for n in num:
        if n < min:
            min = n
    return min

def ft_max(num) -> float:
    if len(num) == 0:
        return float('nan')
    max = num[0]
    for n in num:
        if n > max:
            max = n
    return max

def ft_quartile1(num) -> float:
    num = sorted(num)
    if len(num) == 0:
        print(f"ERROR")
        return float('nan')
    else :
        return num[int(len(num) / 4)]
        print(f"quartile : [{num[int(len(num) / 4)]:.1f}, {num[int(len(num) * 3 / 4)]:.1f}]")

def ft_quartile2(num) -> float:
    num = sorted(num)
    if len(num) == 0:
        return float('nan')
        return float('nan')
    else :
        return num[int(len(num) * 3 / 4)]
        print(f"quartile : [{num[int(len(num) / 4)]:.1f}, {num[int(len(num) * 3 / 4)]:.1f}]")

def ft_percentile(data, perc: int):
    data_sorted = sorted(data)
    n = len(data_sorted)
    if n == 0:
        return float('nan')
    r = perc / 100 * (n - 1)
    k = int(r)
    d = r - k
    
    if k + 1 < n:
        return (1 - d) * data_sorted[k] + d * data_sorted[k + 1]
    else:
        return data_sorted[k]

def ft_variance(num) -> float:
    tot = 0
    i = 0
    var = 0
    for n in num:
        tot += n
        i += 1
    if i == 0:
        return 0
    mean = tot / i
    for n in num:
        var += (n - mean) * (n - mean)
    if i == 1:
        return float("nan")
    var = var / (i - 1)
    return (var)

def ft_standard_deviation(num) -> float:
    var = ft_variance(num)
    if var == 0:
        return float('nan')
    else :
        return sqrt(var)

def ft_statistics(list, func, perc = None) -> float:

    try :
        if func == "count":
            return ft_count(list)
        if func == "mean":
            return ft_mean(list)
        if func == "median":
            return ft_median(list)
        if func == "min":
            return ft_min(list)
        if func == "max":
            return ft_max(list)
        if func == "quartile1":
            return ft_quartile1(list)
        if func == "quartile2":
            return ft_quartile2(list)
        if func == "percentile":
            return ft_percentile(list, perc)
        if func == "std":
            return ft_standard_deviation(list)
        if func == "var":
            return ft_variance(list)
    
    except Exception as e:
        raise e