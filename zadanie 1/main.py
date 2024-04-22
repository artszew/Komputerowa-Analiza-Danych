import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sum_array(arr):
    total = 0.0
    if isinstance(arr[0], (list, tuple)):
        # If the first element of arr is a list or tuple, it's a 2D array.
        for row in arr:
            for element in row:
                total += element
    else:
        # If the first element is not a list or tuple, it's a 1D array.
        for element in arr:
            total += element
    return total


def mean_numbers(arr):
    return float(sum_array(arr))/len(arr)


def std_numbers(arr):
    meann = mean_numbers(arr)
    sum_of_numbers = 0

    for it in range(len(arr)):
        sum_of_numbers += (arr[it] - meann)**2

    return np.sqrt(float(sum_of_numbers)/(len(arr)-1))


def median_numbers(arr):
    arr.sort()
    n = len(arr)
    if n % 2 == 0:
        middle1 = arr[n // 2]
        middle2 = arr[n // 2 - 1]
        return (middle1 + middle2) / 2
    else:
        return arr[n // 2]


def percentile_numbers(arr, percentile):
    arr.sort()
    n = len(arr)
    rank = (percentile / 100) * (n - 1)
    k = int(rank)
    d = rank - k

    if k < 0:
        return arr[0]
    elif k >= n - 1:
        return arr[n - 1]
    else:
        return arr[k]


if __name__ == '__main__':
    data = np.genfromtxt('data.csv', delimiter=',', dtype=float)  # Read data as floats.

    # population of flowers (the number of each type)
    counts = np.bincount(data[:, 4].astype(int))
    tab = np.empty(3, dtype=float)
    for i in range(3):
        tab[i] = round((counts[i] / sum_array(counts)) * 100, 1)

    gatunek = ["setosa", "versicolor", "virginica"]
    data1 = np.empty((4, 2), dtype=object)

    for j in range(3):
        data1[j, 0] = gatunek[j]
        data1[j, 1] = f"{counts[j]}({tab[j]}%)"

    data1[3, 0] = "Razem"
    data1[3, 1] = f"{sum_array(counts)}({np.ceil(sum_array(tab))}%)"

    df1 = pd.DataFrame(data1)
    df1.columns = ['Gatunek', 'Liczebność']
    df1.index = ["" for _ in range(len(df1))]
    print(df1)

    """
    Columns in order:
        Sepal length
        Sepal width
        Petal length
        Petal width
        Species
    """
    cecha = ["Długość działki kielicha", "Szerokość działki kielicha", "Długość płatka",
             "Szerokość płatka"]

    results = np.empty((4, 5), dtype=object)

    for j in range(4):
        results[j, 0] = cecha[j]

    for i in range(4):
        column = data[:, i]

        # Minimum value
        min_value = round(column.min(), 2)
        results[i, 1] = f'{min_value}'

        # Arithmetic average
        mean_value = round(mean_numbers(column), 2)

        # Standard deviation
        std_dev = round(std_numbers(column), 2)
        results[i, 2] = f'{mean_value} (+/-{std_dev})'

        # Median
        median = round(median_numbers(column), 2)
        q1 = round(percentile_numbers(column, 25), 2)
        q3 = round(percentile_numbers(column, 75), 2)

        results[i, 3] = f'{median} ({q1}-{q3})'

        # Maximum value
        max_value = round(column.max(), 2)
        results[i, 4] = f'{max_value}'

    df = pd.DataFrame(results)
    df.columns = ['Cecha', 'Min', 'Śr.aryt. (odch. stan.)', 'Mediana (Q1-Q3)', 'Max']
    df.index = ["" for _ in range(len(df))]
    print(df.to_string())

    # Histograms:
    for i in range(4):
        column = data[:, i]
        plt.figure()
        if i == 0:
            a, b, c = 4, 8, 8
        elif i == 1:
            a, b, c = 2, 4.5, 10
        elif i == 2:
            a, b, c = 1, 7, 12
        else:
            a, b, c = 0, 2.5, 10
        plt.hist(column, bins=c, range=(a, b), color='blue', edgecolor='black')

        plt.title(f'{cecha[i]}', fontsize=24)
        if i % 2:
            plt.xlabel('Szerokość (cm)', fontsize=24)
        else:
            plt.xlabel('Długość (cm)', fontsize=24)
        plt.ylabel('Liczebność', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

    # Boxplotes:
    data = np.genfromtxt('data.csv', delimiter=',', dtype=float)
    for i in range(4):
        column = data[:, i]

        setosa_data = column[data[:, 4] == 0]
        versicolor_data = column[data[:, 4] == 1]
        virginica_data = column[data[:, 4] == 2]
        plt.figure(figsize=(8, 6))

        plt.boxplot([setosa_data, versicolor_data, virginica_data], labels=gatunek)
        if i % 2:
            plt.ylabel('Szerokość (cm)', fontsize=24)
        else:
            plt.ylabel('Długość (cm)', fontsize=24)
        plt.xlabel('Gatunek', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

    plt.show()