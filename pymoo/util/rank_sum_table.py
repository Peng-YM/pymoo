import math

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def rnd(number, significant_digits):
    return round(number, significant_digits - int(math.floor(math.log10(abs(number)))) - 1)


def rank_sum_table(results, columns, rows, baseline=0, goal='minimize', significant_level=0.05, significant_digits=4):
    assert (goal in ['minimize', 'maximize']), "Invalid argument!"
    num_row, num_col, num_observations = results.shape

    # calculate the mean of each item
    mean_values = np.mean(results, axis=2)

    # calculate the standard deviation of each item
    std_values = np.std(results, axis=2)

    # sort the mean value for each row
    orders = np.argsort(mean_values, axis=1)
    orders = np.fliplr(orders) if goal == 'maximize' else orders
    ranks = np.argsort(orders, axis=1)

    # rank sum test
    counts = np.zeros((3, num_col), dtype=np.int)  # count of +/-/=
    table = pd.DataFrame(columns=columns, index=[*rows, "+/-/\u2248"])
    for i in range(num_row):
        base = results[i, baseline]
        for j in range(num_col):
            current = results[i, j]
            [_, p_value] = ranksums(current, base)
            m, s, r = rnd(mean_values[i, j], significant_digits), rnd(std_values[i, j], significant_digits), ranks[i, j]
            table.iloc[i, j] = f"{m} \u00B1 {s} ({r})"
            if j == baseline:
                continue
            if p_value < significant_level:
                if (m < mean_values[i, baseline]) ^ (goal == 'minimize'):
                    counts[1, j] += 1  # current observations are worse than the baseline --> -
                    table.iloc[i, j] += " -"
                else:
                    counts[0, j] += 1  # current observations are better than the baseline --> +
                    table.iloc[i, j] += " +"
            else:
                # no significant different between current observation and the baseline
                counts[2, j] += 1
                table.iloc[i, j] += " \u2248"
    # the summary line
    for j in range(num_col):
        table.iloc[num_row, j] = 'baseline' if j == baseline else (f"{counts[0, j]}/{counts[1, j]}/{counts[2, j]}")

    return table


if __name__ == '__main__':
    columns = ["Algorithm 1", "Algorithm 2", "Algorithm 3"]
    rows = ["Problem 1", "Problem 2", "Problem 3"]
    n_observations = 31
    results = np.zeros((len(rows), len(columns), n_observations))
    # randomly generate some observations
    for i, _ in enumerate(columns):
        for j, _ in enumerate(rows):
            noise = np.random.standard_normal(n_observations)
            results[i, j] = noise + (3 if i == j else 1)

    # perform the rank sum test
    table = rank_sum_table(results, columns, rows, goal='maximize')
    print(table)
