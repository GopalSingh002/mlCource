import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sns.set()
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (11,8)

DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"+"mlbootcamp5_train.csv"
df = pd.read_csv(DATA_PATH, sep=";")
# print(df.info())

df_uniques = pd.melt(frame=df, value_vars=["gender", "cholesterol", "gluc", "smoke", "alco", "active"],id_vars="cardio")

df_uniques = (pd.DataFrame(df_uniques.groupby(["variable", "value", "cardio"])["value"].count()).sort_index(level=[0, 1]).rename(columns={"value": "count"}).reset_index())

# print(df_uniques.info())
# print(df_uniques.head())

# sns.catplot(x="variable", y="count", hue="value", data=df_uniques, kind="bar", col="cardio")
# plt.xticks(rotation='vertical')
# plt.show()

"""
Printing each column unique values and their counts
"""

# for c in df.columns:
#     print("Column: ", c)
#     n = df[c].nunique()
#     print("Number of unique values: ", n)
#     if n<=3:
#         print("unique values and their counts: ",sorted(df[c].value_counts().to_dict().items()))

# print(df.info())
"""
Assignment questions:
"""




# 1.1 How many men and women are present in this dataset? Values of the gender feature were not given (whether “1” stands for women or for men) 
# figure this out by looking analyzing height, making the assumption that men are taller on average.

# print(df["gender"].unique())
# fig1 = px.box(df, x="gender", y="height")   
# fig1.show()
"""
After ploting the graph, we analyzed that the median of gender 2 is 170 and gender 1 is 161

so gender 2 = male, 1 = female
"""
# print(df['gender'].value_counts())
"""
this statement returns: and we can conclude:
45530 females and 24470 males
"""




# 1.2. (1 point). Who more often report consuming alcohol – men or women?
# sns.countplot(x="alco", hue="gender", data=df)
# plt.show()

"""
This count plot conclude that the men are more often reported consuming alcohol
"""



# 1.3 What’s the rounded difference between the percentages of smokers among men and women?
# print(np.round(df.groupby("gender")["smoke"].value_counts(normalize=True)*100,0))
"""
This will conclude that
rounded % of 
female smoker: 2
male smoker: 22
"""


# 1.4 What’s the rounded difference between median values of age (in months) for non-smokers and smokers?
# You’ll need to figure out the units of feature age in this dataset.
# df["age"] = df["age"]/30
# print(np.round(df[df["smoke"]==0]["age"].median()-df[df["smoke"]==1]["age"].median(),0))

# fig3 = px.box(df, x="smoke", y="age")
# fig3.show()
"""
In the box plot we can analyze the median for smoker and non smoker
non-smoker : 657.4
smoker : 637.5

rounded difference: 657.5-638.5 = 20
"""


#1.5 Calculate fractions of ill people (with CVD) in the two groups of people described in the task. What’s the ratio of these two fractions?
df["age"] = np.round(df["age"]/365.25, 0).astype(int)

df = df[(df["age"] >=60) & (df["age"]<65) &(df["gender"] == 2) &(df["smoke"] == 1)]


a = df[(df["cholesterol"] == 1) & (df["ap_hi"] <120)]["cardio"].mean()

b = df[(df["cholesterol"] == 3) & (df["ap_hi"]>=160)&(df["ap_hi"]<180)]["cardio"].mean()
print(f"Ration of fraction {np.round(a,2)} ,{np.round(b,2)}: ", np.round(a/b, 2))


 
"""
Question 1.6. (2 points). Choose the correct statements:                    Ans

Median BMI in the sample is within boundaries of normal values. :           26.4

Women’s BMI is on average higher then men’s.                               1: 26.7 , 2: 25.9  

Healthy people have higher median BMI than ill people.                     healthy: 25.5, Ill: 27.5             

In the segment of healthy and non-drinking men BMI is closer to the norm than in the segment of healthy and non-drinking women
"""

# # print(df.info())
# df["BMI"] = np.round(df["weight"]/(df["height"]/100)**2,1)
# # print(df.head())
# # fig4 = px.box(df, x = "BMI")
# # fig4.show()

# # fig5 = px.box(df,x = "gender", y = "BMI")
# # fig5.show()

# # fig6 = px.box(df, x= "cardio", y="BMI")
# # fig6.show()

# menDf = df[(df["gender"] == 2) & (df["cardio"] == 0) &(df["alco"] == 0)]
# womenDf = df[(df["gender"] == 1) & (df["cardio"] == 0) &(df["alco"] == 0)]

# print("Men: ", menDf["BMI"].mean())
# print("Women: ", womenDf["BMI"].mean())


"""
1.7 What percent of the original data (rounded) did we filter out in the previous step?
"""
# before = df.shape[0]
# df = df[(df["ap_lo"] > df["ap_hi"])| (df["height"] < df["height"].quantile(0.025)) | (df["height"] > df["height"].quantile(0.975)) | (df["weight"] < df["weight"].quantile(0.025)) | (df["weight"] > df["weight"].quantile(0.975))]
# after = df.shape[0]
# print(np.round(after/before*100,0))
# # print(df.info())

"""
2.1. Which pair of features has the strongest Pearson’s correlation with the gender feature?

Cardio, Cholesterol

Height, Smoke

Smoke, Alco

Height, Weight
"""
# corr = df.corrwith(df["gender"], method='pearson', axis=0)*100
# print(corr)


"""
Task: Height distribution of men and women
Create a violin plot for the height and gender using violinplot(). Use the parameters:

hue to split by gender;

scale to evaluate the number of records for each gender.

In order for the plot to render correctly, 
you need to convert your DataFrame to long format using the melt() function from pandas.

"""


# vio_df = pd.melt(df, value_vars=["height", "weight"], id_vars="gender")
# sns.violinplot(x= "variable", y="value", hue="gender", data=vio_df, scale='count')
# plt.show()




"""
task: Rank correlation

Calculate and plot a correlation matrix using the Spearman’s rank correlation coefficient

2.2. Which pair of features has the strongest Spearman rank correlation?

Height, Weight: 0.47

Age, Weight : 0.01

Cholesterol, Gluc : 0.4

Cardio, Cholesterol : 0.22

Ap_hi, Ap_lo : 0.72

Smoke, Alco : 0.33

# all the values are analyzed from the heat map





2.3. Why do these features have strong rank correlation?

Inaccuracies in the data (data acquisition errors).

Relation is wrong, these features should not be related.

Nature of the data.



"""
# corr1 = np.round(df.corr(method="spearman"),2)
# sns.heatmap(corr1, annot=True)
# plt.show()



"""
Question 2.4. (1 point). What is the smallest age 
at which the number of people with CVD outnumbers 
the number of people without CVD?

44

55

64

70
"""



# df["age"] = np.round(df["age"]/365.25,0).astype(int)

# sns.countplot(data=df, x="age", hue="cardio")
# plt.show()


# first age group of cardio who surpass not cardio people



"""
Assignment final score: 18/18

"""