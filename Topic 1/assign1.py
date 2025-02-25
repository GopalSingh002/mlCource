import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
df = pd.read_csv(DATA_URL + "adult.data.csv").copy()
# print(df.info())
# 1. How many men and women (sex feature) are represented in this dataset?
# print(df["sex"].value_counts())

# 2. What is the average age (age feature) of women?
# print(np.round(df[df['sex'] == "Female"]["age"].mean(), 2))

# 3. What is the percentage of German citizens (native-country feature)?
# print(df["native-country"].value_counts(normalize=True)*100)

# 4-5. What are the mean and standard deviation of age for those who earn more than 50K per year (salary feature) and those who earn less than 50K per year?
# print(np.round(df[df["salary"]=="<=50K"]["age"].describe(), 0))


# 6. Is it true that people who earn more than 50K have at least high school education? (education – Bachelors, Prof-school, Assoc-acdm, Assoc-voc, Masters or Doctorate feature)
# print(df[(df["education"] == "HS-grad")&(df["salary"] == ">50K")])
# print(df[df["salary"] == ">50K"].value_counts())

# 7. Find the maximum age of men of Amer-Indian-Eskimo race.
# print(df[df["race"] == "Amer-Indian-Eskimo"]["age"].max())

# 8. Among whom is the proportion of those who earn a lot (>50K) greater: married or single men (marital-status feature)? 
# Consider as married those who have a marital-status starting with Married 
# (Married-civ-spouse, Married-spouse-absent or Married-AF-spouse), the rest are considered bachelors.
# df["marital-status"] = df["marital-status"].apply(lambda status:'Married' if status == "Married-civ-spouse" or status == "Married-spouse-absent" or status == "Married-AF-spouse" else 'Single')
# print(df.groupby("salary")["marital-status"].value_counts(normalize=True))

# 9. What is the maximum number of hours a person works per week (hours-per-week feature)? 
# How many people work such a number of hours, and what is the percentage of those who earn a lot (>50K) among them?
# print(df["hours-per-week"].max())
# print(df[df["hours-per-week"] == 99]["hours-per-week"].count())
# print(df[(df["hours-per-week"] == 99)]["salary"].value_counts(normalize=True))

# 10. Count the average time of work (hours-per-week)
# for those who earn a little and a lot (salary) for each country (native-country). What will these be for Japan?
# print(df[df["native-country"] == "Japan"].groupby("salary")["hours-per-week"].mean())



"""
Final score = 10/10
"""

