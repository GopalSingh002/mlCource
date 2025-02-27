import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
df = pd.read_csv("https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/" + "telecom_churn.csv")
# print(df.info())

"""
Types of visualization:
------------------------------------------------------------------------------------------------------------------------------------
1. Univariate visualization
    In this visualization we focused on a single feature, instead of comparing it to others we focused on the how the values are distributed of that feature.

Types of univariate visualization:
a. Quantitative features: this features take an orderd numerical values. Values that are real, continous and usually express a count or measurements.
    -hist() and density() plots

:-hist(): 
    histograms are the easiest way to distribute numeric variables.
    histogram group the values into bins of equal value range. The shape of the histogram may contians clues like about the underlying distribution
    like gaussins and exponantial. We can also find the skewness in the plot by seeing the histogram
    skewness due to the regular distribution but has some anomalies. 

Use case:
    - best for understand distribution of numerical data
    - Identfy skeweness, peak, gaps in data
    - well for large data set


    ex:
Syntax:
features = [] that you are going to display
df[feature/column].hist(figsize = (x,y));

- we can create hist for each manually also we can just pass the list of the column or features in a list and it create hist for each

- in below example we can see that the Total day minutes graph is normally distributed, but Total inlt calls are skewed towards right  
    """

features = ["Total day minutes", "Total intl calls"]
# df[features].hist(figsize=(10,5))
# plt.show()


"""
b. density()

now if we need to tsee the distribution in more clear way, we can use the density graph or more formal kernal density graph
this is the smoothen version of the histogrm graph.
It doesn't depands on the size of bins.

- smoothen version of hist
- usefull where you want a continous, smooth estiates of distribution
- help to visualize multiple distribution on the same plot.

like comparing age distribution of two gender
uderstanding probability densities of continous features
"""
# df[features].plot(kind="density", subplots=True, sharex=False, layout=(1,2) ,figsize=(10,5))
# plt.show()

    # or

# sns.kdeplot(data=df["Total day calls"])
# plt.show()

"""
instead of using pandas visualization tools we can go with seaborn like:

in this the hight of the histo bar represent the norm and show the density not the values o example in each bin
"""

# sns.histplot(df['Total day calls'], stat='density', kde=True) # till now, it only show the hist graph   # it will add the kernal density on the graph so that we can spot the skeweness and value distribution graph
# plt.show()

"""
c. Box plot

Syntax: 
sns.boxplot(x="feature", data=df)

lets see how to interpret  the box plot first:
-its components are box and are called whiskers and the number of individual points are outliers

- the box itself illustrates the interquartile spread of the distribution.

- lenght is deter by 25th qt and 75qt %.

- vertical line inside the box is called median = 50%

- the whiskers are the lines extending from the box. they represent entire scatter of data points, specially points fall withing the interval

(Q1-1.5 * IQR, Q3+1.5 * IQR)
where IQR = Q3-Q1 is the interquartile range

- Box:
    Q1 : First hori line, 25%
    Q2 : Median , 50%
    Q3: Last Hori line, 75%
    IQR : Q3-Q1

    Lower whisker :  1.5*IQR = Q1-1.5*IQR
    upper Whisker: 1.5*IQR = Q3+1.5*IQR

- Outliers that fall outside of the range bounded by the whiskers are plotted individually as black point along the central axis

Use case:
- Best for detect outliers and comparing distribution
- show median, quartiles, extreme values
- work well for comparing multiple groups

sns.boxplot(x = feature1, y= feature2, data=df)

"""
# sns.boxplot(x="Total day calls", data=df)
# plt.show()


"""
d. Violin plot

it have the kernal density estimate on both sides

diff b/w box and violin:
- box plot illustration certain statistics concerning individual examples of dataset, while the violin plot concentrates more on smoothen distribution as whole

Syntx:
sns.violinplot(data = df[feature], ax=axis)
axis= horizontal or vertical
        0               1

Use case:
- Combination of box plot + density plot
- provide detailed distribution shape while keeping summary stat
- more info than box


ex:
Comparing income distribution across job roles
        
"""

# sns.violinplot(data=df["Total day calls"])
# plt.show()


"""
------------------------------------------------------------------------------------------------------------------------------------
2. Categorical and binary features
categoricl feature: take fixed number of values, each of this value corresponding to a group.
Binary features: when the number of possible values are 2.

- if the categorical data is ordered: ordinal

Types:
a. Frequancy table : value_counts(): which provide us the group of values and their count.
    by default the entries in the output are sorted from the most to least frequantly occuring values

    
b. Bar plot
- Graphical representation of the frequancy table.
- seaborn gives two function for this:
    countplot()
    barplot()

    Syntax:
    sns.countplot(x="feature", data=df, ax)

confusion is like hist and bar may look similar

- hist suited for distribution of num variables, while bar plot suited for categorical features
- hist x-axis: numerical, bar x-axis: num, string, bool
- hist x-axis: cartesian coordinate along which value cannot be changed
while in bar plot, ordering of the bar not predicted.

- but the bars are sorted by their heights

"""
"""
Countplot:
-used to count the occurance of categries in categ column
-similar to hist but for categorical data
- Only takes on categorical varibale, x or y

use case:
-count num of a particular group or groups in dataset

args:
-x or y : categorical var
data = : from which using x or y
hue: adds a second categorical variable
palette :  colour
order: custom order for categories


"""
# sns.countplot(x="Area code", data=df, hue="Churn")
# plt.show()

"""
bar plot:
- used for stat aggregation (mean, sum, max etc)
-  show num value for each categ

use case:
- finding avrg
- comparing avrg values per type
- checking total value per category

args:
x or y : x(categorical value) and y(numerical value)
hue: grouping another variable
estimator: function applied 
palette: color

"""

# sns.barplot(x="Churn",y ="Total day calls", data=df, hue="Area code")
# plt.show()


"""
------------------------------------------------------------------------------------------------------------------------------------
3. Multivariate visualization
this plots allow us to see the relationship between two or more different variables in one fig.

Types:
a. Quantitative Vs Qauntitative
- when both varibales are numeric

I. correlation matrix:
- It show how two numerical variables are related using correlation value(-1 to 1)

Ex checking is there any relation between feature1 and feature 2 

"""
numerical = list(set(df.columns) - set(["State", "International plan", "Voice mail plan", "Area code", "Churn","Customer service calls",]))
# print(df[numerical].corr())


"""
II. Scatter plot:
How two numerical variables relates using dots
ex: Total inlt calls Vs Total intl charge

we can do this in two ways:
- sns.scatterplot(x,y,data)

-sns.jointplot(x,y,data,kind='scatter) # it join the hist with the scatter
"""
# sns.scatterplot(x = "Total intl charge", y="Total intl calls", data=df)
# plt.show()

# sns.jointplot(x="Total intl charge", y="Total intl calls", data=df, kind='scatter')
# plt.show()


"""
We can actually change the colour of two categories so that it is easier to find the different value on the scatter graph to identify each one
by mapping it 

like category: Yes, No
sns.scatterplot(x, y, data= df, palette = "", hue=col)
you need to pass the hue for grouping only then the palette work

"""
# sns.scatterplot(x = "Total intl charge", y="Total intl calls", hue="Churn",data=df, palette="Set2")
# plt.show()

"""
III. Scatterplot matrix
-pairplot()
Plot scatter plot for all numerical values at once
ex: if we have three numerical variables(total intl charge, calls, minutes)
, it will show all possible plots

"""
# sns.pairplot(df[["Total intl calls", "Total intl charge", "Total intl minutes"]])
# plt.show()


"""
b. Quantitative Vs Categorical  
-For one numerical and one categorical
- here we use linear models
lmplot()
- Fits a regression line to see trends between a category and a number

ex: checking total intl calls depends on churn , hue is optional, it use groupby property for result
"""
# sns.lmplot(x ="Total intl calls", y="Churn", data=df, hue="International plan")
# plt.show()


"""
c. Categorical Vs Categorical
- When both are categorical

I. Grouped countplot()
- count the occurance of two categories together
counting how many area code belong to each Churn
"""

# sns.countplot(x = "Area code", data=df, hue="Churn")
# plt.show()

"""
II. Contigency table:
-creating pivot table
"""
# print(pd.crosstab(df["Area code"], df["Churn"]))


"""
------------------------------------------------------------------------------------------------------------------------------------
4. Whole dataset visualization
a dataset contains n number of features. We need to look at each feature how?

a. naive approach:
- Basic visualization
- Uses basic plots like hist, scatter to view all variables at once.
-Problem: If we have 100+ variable, this method became slow

"""
# df.hist(figsize=(10,10)) # create hist graph for each feature
# sns.pairplot(df)
# plt.show()


"""
b. Dimension reduction:
-What if we have lot of features. We try to reduce the dimension of the dataset, without loosing much info.
- Most widely used dimension reduction method is PCA
- Principle Component Analysis
- It reduce many variables into a few key ones that explain most of the data
- for example, if a dataset contains 50 columns, it can reduce it upto 2-3 columns that exlain upto 90% of data
for PCA we need to use scikit-learn =sklearn module
sklearn.decomposetion import PCA

"""

# pca = PCA(n_components=2)
# dfAfterPca = pca.fit_transform(df[numerical])
# print("Original Dataframe: \n",df)
# print("Reduced dataframe: \n", dfAfterPca)
# print("reduced dimension: ", dfAfterPca.shape)

"""
c. t-SNE
- Visualize the data in 2d or 3d
- Unlike PCA which is linear, t-SNE captures complex patterns
- we required:
sklearn.manifold import TSNE

"""
# tsne = TSNE(n_components=2)
# dfAfterTSNE = tsne.fit_transform(df[numerical])
# print("Original Dataframe: \n",df[numerical])
# print("Reduced dataframe by TSNE: \n", dfAfterTSNE)
# print("reduced dimension: ", dfAfterTSNE.shape)



