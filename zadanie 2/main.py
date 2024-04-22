import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# Sum
def sum_array(arr):
    total = 0.0
    for element in arr:
        total += element
    return total


# Mean
def mean_numbers(arr):
    if len(arr) == 0:
        raise ValueError("Input array is empty.")
    return sum_array(arr) / len(arr)


# Pearson correlation coefficient
def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    mean_x = mean_numbers(x)
    mean_y = mean_numbers(y)

    numerator = sum_array([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    denominator_x = np.sqrt(sum_array([(x[i] - mean_x) ** 2 for i in range(len(x))]))
    denominator_y = np.sqrt(sum_array([(y[i] - mean_y) ** 2 for i in range(len(y))]))

    return numerator / (denominator_x * denominator_y)


# Linear regression using the least squares method
def linear_regression(x, y):
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")

    mean_x = mean_numbers(x)
    mean_y = mean_numbers(y)

    numerator = sum_array([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
    denominator = sum_array([(x[i] - mean_x) ** 2 for i in range(len(x))])

    a = numerator / denominator
    b = mean_y - a * mean_x

    return a, b


if __name__ == '__main__':
    data = np.genfromtxt('data.csv', delimiter=',', dtype=float)  # Read data as floats.

    # Extracting four columns for analysis
    feature1 = data[:, 0]
    feature2 = data[:, 1]
    feature3 = data[:, 2]
    feature4 = data[:, 3]

    # Calculate and round Pearson correlation coefficients
    p_12 = round(pearson_correlation(feature1, feature2), 2)
    p_13 = round(pearson_correlation(feature1, feature3), 2)
    p_14 = round(pearson_correlation(feature1, feature4), 2)
    p_23 = round(pearson_correlation(feature2, feature3), 2)
    p_24 = round(pearson_correlation(feature2, feature4), 2)
    p_34 = round(pearson_correlation(feature3, feature4), 2)

    # Linear regression
    a_12, b_12 = linear_regression(feature1, feature2)
    a_13, b_13 = linear_regression(feature1, feature3)
    a_14, b_14 = linear_regression(feature1, feature4)
    a_23, b_23 = linear_regression(feature2, feature3)
    a_24, b_24 = linear_regression(feature2, feature4)
    a_34, b_34 = linear_regression(feature3, feature4)

    # Pairs of features
    c_12 = "długość działki x szerokość działki"
    c_13 = "długość działki x dlugosc platka"
    c_14 = "długość działki x szerokosc platka"
    c_23 = "szerokosc dzialki x dlugosc platka"
    c_24 = "szerokosc dzialki x szerokosc platka"
    c_34 = "dlugosc platka x szerokosc platka"

    # Printing resulsts in the data frame
    results = [
        [c_12, p_12, f"y = {round(a_12, 1)}x+{round(b_12, 1)}"],
        [c_13, p_13, f"y = {round(a_13, 1)}x{round(b_13, 1)}"],
        [c_14, p_14, f"y = {round(a_14, 1)}x{round(b_14, 1)}"],
        [c_23, p_23, f"y = {round(a_23, 1)}x+{round(b_23, 1)}"],
        [c_24, p_24, f"y = {round(a_24, 1)}x+{round(b_24, 1)}"],
        [c_34, p_34, f"y = {round(a_34, 1)}x{round(b_34, 1)}"],
    ]
    df_results = pd.DataFrame(results, columns=["Para cech", "r(Pearson)", "Równanie regresji:"])
    print(df_results)

    # Plotting scatter plots and regression lines
    plt.figure(figsize=(12, 8))

    # Scatter plots with regression lines
    plt.subplot(2, 3, 1)
    sns.regplot(x=feature1, y=feature2, line_kws={'color': 'red'}, ci=None)
    plt.title("r = -0.12; y = -0.1x+3.4", fontsize=14)
    plt.xlabel("Długość działki (cm)", fontsize=14)  # Zwiększono rozmiar etykiety osi x
    plt.ylabel("Szerokość działki (cm)", fontsize=14)  # Zwiększono rozmiar etykiety osi y

    plt.subplot(2, 3, 2)
    sns.regplot(x=feature1, y=feature3, line_kws={'color': 'red'}, ci=None)
    plt.title("r = 0.87; y = 1.9x-7.1", fontsize=14)
    plt.xlabel("Długość działki (cm)", fontsize=14)
    plt.ylabel("Długość płatka (cm)", fontsize=14)

    plt.subplot(2, 3, 3)
    sns.regplot(x=feature1, y=feature4, line_kws={'color': 'red'}, ci=None)
    plt.title("r = 0.82; y = 0.8x-3.2", fontsize=14)
    plt.xlabel("Długość działki (cm)", fontsize=14)
    plt.ylabel("Szerokość płatka (cm)", fontsize=14)

    plt.subplot(2, 3, 4)
    sns.regplot(x=feature2, y=feature3, line_kws={'color': 'red'}, ci=None)
    plt.title("r = -0.43; y = -1.7x+9.1", fontsize=14)
    plt.xlabel("Szerokość działki (cm)", fontsize=14)
    plt.ylabel("Długość płatka (cm)", fontsize=14)

    plt.subplot(2, 3, 5)
    sns.regplot(x=feature2, y=feature4, line_kws={'color': 'red'}, ci=None)
    plt.title("r = -0.37; y = -0.6x+3.2", fontsize=14)
    plt.xlabel("Szerokość działki (cm)", fontsize=14)
    plt.ylabel("Szerokość płatka (cm)", fontsize=14)

    plt.subplot(2, 3, 6)
    sns.regplot(x=feature3, y=feature4, line_kws={'color': 'red'}, ci=None)
    plt.title("r = 0.96; y = 0.4x-0.4", fontsize=14)
    plt.xlabel("Długość płatka (cm)", fontsize=14)
    plt.ylabel("Szerokość płatka (cm)", fontsize=14)

    plt.tight_layout()
    plt.show()
