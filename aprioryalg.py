import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

data = pd.read_csv("TV.csv")

transactions = []
for i in range(data.shape[0]):
    transactions.append([str(data.values[i,j]) for j in range(data.shape[1])])
print(transactions)

minsup = [1, 3, 5, 10, 15]
rules = apriori(transactions=transactions, min_support=0.001, min_cinfidence=0.02, min_lift=3, min_length=2, max_length=2)
results = list(rules)
print(len(results))

def lift(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsDF = pd.DataFrame(lift(results), columns=["Left", "Right", "Support", "Confidence", "Lift"])
print(resultsDF.nlargest(n = 10, columns = "Lift"))

# Результаты для первого набора (TV)
number = [1529, 717, 430, 156, 71]
plt.bar(minsup, number)
plt.xlabel('MinSup, %', fontsize=12)
plt.ylabel('Number of transactions', fontsize=12)
plt.show()

time = [32, 28, 22, 15, 7]
plt.bar(minsup, time)
plt.xlabel('MinSup, %', fontsize=12)
plt.ylabel('Time, sec', fontsize=12)
plt.show()

# Результаты для второго набора (retail)
number = [686, 341, 243, 94, 47]
plt.bar(minsup, number)
plt.xlabel('MinSup, %', fontsize=12)
plt.ylabel('Number of transactions', fontsize=12)
plt.show()

time = [21, 17, 14, 8, 4]
plt.bar(minsup, time)
plt.xlabel('MinSup, %', fontsize=12)
plt.ylabel('Time, sec', fontsize=12)
plt.show()