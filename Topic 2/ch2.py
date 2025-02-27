import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
sns.set()

"""
Setting the dataset for the learning
"""
df = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"+"video_games_sales.csv")
df = df.dropna() #removing all the rows that have null values


df1 = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/" + "telecom_churn.csv")
df1 = df1.dropna()

#Converting object type for the ease
df["User_Score"]= df["User_Score"].astype("float64")
df["Year_of_Release"] = df["Year_of_Release"].astype("int64")
df["User_Count"] = df["User_Count"].astype("int64")
df["Critic_Count"] = df["Critic_Count"].astype("int64")

# print(df.info())
# print(df.head())

cols = [            # we are going to use only these columns in this learning
    "Name",
    "Platform",
    "Year_of_Release",
    "Genre",
    "Global_Sales",
    "Critic_Score",
    "Critic_Count",
    "User_Score",
    "User_Count",
    "Rating",
]
df =df[cols]
# print(df.info())


"""
Correlation
 we can do this with one more way:
 correlation of while dataset with a particular feature
- there should be no string feature and the passing arg must a pandas series
"""
# df.corrwith(df['Total day minutes'])  # before running this, ensure there is no string value containing feature


"""
1. DataFrame.plot()
- in pandas, we have the feature of visualization, which is built on matplotlib

Syntax:
df.plot(kind="type", x="col", y = "col",figsize(width, height), title="")

kind:- type of plot
x:- column for x, y:- colunm for y
figsize:- size of the figure
title:- title of the plot

Types:
line: default, use for trend over time
bar: vertical bar for categorical data
barh: same as bar but horizontal
hist: show distribution of data
box: show summary stat(median, quartile, outliers)
scatter: used to show relation between numerical var
area :filled area under the line 
pie: used for cat proportion
"""

# df.plot(kind='line', x="Global_Sales", y="Year_of_Release")
# df.plot(kind='bar', x="Year_of_Release", y="Global_Sales")
# df.plot(kind='barh', x="Year_of_Release", y="Global_Sales", figsize=(10,10))
# df["User_Count"].plot(kind='hist', bins=20)
# df["User_Count"].plot(kind='box')
# df.plot(kind='scatter', x="Year_of_Release", y="Global_Sales")
# df.plot(kind='area', x="User_Count", y="Global_Sales")
# df["Year_of_Release"].plot(kind='pie', figsize=(10,10))


# plt.show()  #if you have to run for all then paste this after each plot


"""
2. Seaborn

a. pairplot():
it pair different variables.Which create a matrix of scatte plots by default. This kind of plot helps us visualize the relationship between different variables in a single output
"""

# sns.pairplot(df[["Global_Sales", "User_Count", "User_Score"]])
# plt.show()

"""
b. histplot():
It help use to see the skeweness of the values and their distribution
- kde: kernal density estimation, which add a curve on the hist.

"""
# sns.histplot(df["Critic_Count"], kde=True, stat='density')
# plt.show()


"""
c. jointplot():
it join the hist graph in the scatter graph or other graph
helpful for look closer relationship between two numerical variables
this is a cross between a scatter and hist
"""
# sns.jointplot(x= "User_Count", y="User_Score", data=df, kind="scatter")
# plt.show()

"""
d. boxplot():
it help to determine the outliers, whiskers, quartiles

whiskers are those who are out of box, lower and upper
orient='h': to make it horizontal, by default vertical
"""
# sns.boxplot(x="Critic_Score", y = "Platform", data=df)
# plt.show()


"""
e. heatmap()
heatmap destribute, a numerical value over two categorical variables
-It use colours to represent the data. it help to identify the correlation, patterns and outliers
ex:
cat = Platform, Genre
Value = Global_Sales

arg:

data: must 2d, correlation matrix, pivot table
annot = True: Display the value inside the cell
cmap = "coolwarm": Set the color theme("viridis", "Blues", "magma)
vmin = min scale value default will be the lowest value of the set
vmax: max scale value on heat scale, default will be max value on the set
"""
# piv = df.pivot_table(index="Platform", columns="Genre", values="Global_Sales")
# sns.heatmap(piv, annot=True, cmap='magma')

# corrMatrix = df[["Global_Sales", "User_Count", "User_Score", "Year_of_Release", "Critic_Score", "Critic_Count"]].corr()
# sns.heatmap(corrMatrix, annot=True, cmap='magma')
# plt.show()




"""
f. catplot()
used for categorical data visualization.
it provide multi-plot grid for different type of categorical plots like bar, box, strip, violin etc

args:
x= cat variable
y= num variable
data = dataframe containing data
kind='strip, swarm, box, violin, bar, count,
hue= split the data into multiple cat
col= create subplots based on a column
row= create subplots based on a row

"""


"""
3. Plotly
It is an open source library that allows creation of interactive plots
- They provide user interface for detailed data exploration.
ex: You can see exact numerical values by mousing over points.
hiding uninteresting series from the visualization, zoom in onto a specific part of the plot


--------------------
for pyplot we need some libraries
import plotly.express as allias

1. Box plot:
Box plot show the distribution of dataset, highlighting the median, quartiles, and outliers
df
x = category
y = values
"""
# fig = px.box(df, x="Platform", y = "Global_Sales")
# fig.show()

# fig1 = px.box(df1, x="Total day minutes", title="Total day minutes")
# fig1.show()

"""
2. Bar chart
It used to compare values between different categories
df
x = category
y = values
color="col"  if not given same colour for each bar, else different color for that category bars
"""
# fig = px.bar(df, x="Platform",y= "Global_Sales", color="Global_Sales")  # here the bars are colour based on the value intensity, higher get darker or lower get lighter. do this when magnitude of values matters more than categories
# fig.show()


# fig = px.bar(df, x = "Platform", y="Global_Sales", color="Platform")  # now each category get different colours
# fig.show()

"""
3. Line plot

Trend overtime
df
x = year
y = values
color="col"  if not given same colour for each bar, else different color for that category bars
"""
# df1 = pd.DataFrame({
#     "Year":[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
#     "Score":[1342,1012, 1231, 1224, 1454, 1221, 1231, 2101, 1001, 2110, 1202, 1210, 2119, 3432,1212],
#     "Count":[500, 457, 423, 432, 232, 234, 254,123, 214, 153, 164, 120, 10, 164, 189],
# })
# fig = px.line(df1, x="Year", y =["Score", "Count"])
# fig.show()




"""
Extra:

Plotting graph using TSNE , a dimension reduction tech
"""
# normalize the df1 so that each column contains numeric value
# df1.dropna(inplace=True)
# df1.drop("State", axis=1, inplace=True)
# df1["International plan"] = df1["International plan"].map({"Yes":1, "No":0})
# df1["Voice mail plan"] = df1["Voice mail plan"].map({"Yes":1, "No":0})

# tsne = TSNE(random_state=42)
# reprr = tsne.fit_transform(df1)
# plt.scatter(reprr[df1["Churn"] == 1, 0], reprr[df1["Churn"] == 1, 1], c="Blue", label = "Churned")
# plt.scatter(reprr[df1["Churn"] == 0, 0], reprr[df1["Churn"] == 0, 1], c="Orange", label="Non Churned")
# plt.xlabel("Axis - x")
# plt.ylabel("Axis - y")
# plt.legend()
# plt.show()
# print(reprr)

"""
to save the fig in png
plt.savefig(path, dpi)
path: location and name
dpi: resolution

"""



"""
Masking the correlation matrix upper triangle:

corr = df.corr()
for upper triangle hidden : np.triu():
-mask = np.triu(np.ones_like(corr, dtype=bool))

for lower triangle hidden: np.tril()
-mask = np.tril(np.ones_like(corr, dtype=bool))


np.ones_like(corr, dtype=bool): Creates a matrix of True values with the same shape as the correlation matrix.

np.triu(...): Keeps only the upper triangle as True, making it a mask.

-mask=mask: Pass this mask into sns.heatmap() to hide the upper triangle.

"""
