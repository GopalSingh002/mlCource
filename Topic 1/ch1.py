"""
Pandas:------------------------------------------------------------------

Series: A collection or Array of same datatype.
Dataframe: A collection of series.

row: Each row represents a instance.
Column: Each column represents the feature of that instance.

"""


"""
Functions--------------------------------------------------------------------
1. set_option(): in pandas this function is used to change the settings of the dataframe globally like: formatting, display width, precisions
ex:
pd.set_option('display.precision', 2): for the data frame, all the floating values will be displayed till 2 decimal places


2. pd.read_csv(url, skipinitialspace = True)
: to read the csv file, skipinitialspace remove the whitespaces from the start-end of each value

3. dataframe.copy() : copy the dataframe

4. df.isnull(): return the count of values in datasets where the value is null in the whole dataset not a list
--we can do this on a specific column also, df[column name].isnull()

5. df.shape : return the shape/dimensions of the dataframe

6. df.columns : this will return the list of columns also we can assign the new column names with this instruction



"""
#importing pandas in program
import pandas as pd
import numpy as np


#saving the url for the dataset
dataset = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/" + "telecom_churn.csv"

pd.set_option("display.precision", 2)   # this will change the precision setting for the pandas, and display the float element upto two decimal places.


# Creating the dataFrame for the telecom churn dataset
churn = pd.read_csv(dataset, skipinitialspace = True)


#We create a copy of dataset/dataframe so that we can backtrack to the original dataset without any problem
df = churn.copy()


"""
df.head(): for the description of the dataframe

description contains:
count for each colunm(count only valid rows , no NaN will be counted)
mean of the data
standard daviation
min value
max value
25% value from which the all values are less than value = 25%
50% same
75% same
max value 



but this only provide the numerical stats, for non numeric, we tell the function explicitly by passing:
include-
ex:
df.describe(include=["object", "string", "bool"])

- for categorical type: object and bool type, we can use value_counts() function

value_count() for counting
- if we pass normalize = True in this function, it will print the fraction values

"""
# print(df.describe())
# print(df.describe(include=["object", "bool", "string"]))
# print(df["Churn"].value_counts())   #this line return the count for bool false, true in col churn
# print(df["Churn"].value_counts(normalize=True))
# here we can se that the values are too large, we can do one thing. we can change the precision for the results


# pd.set_option('display.precision', 2) # this will change the settings for the pd.
print("values:")
print(df["Churn"].value_counts(normalize=True)) # now the value are two decimal




"""
If we want to change the columns we can use

df.columns = [list containing name of each column] //must be equal to the number of columns in dataframe

let's use this on churn column

"""


"""
Accessing the specific columns

dataframe['columnsname']
"""


"""
Like instead of getting data as whole, we can get the data as slicing feature
like accessing the part of the data

-df.iloc[fromrow : torow, fromcolumn:tocolumn]
here n means which column
iloc[0:4, 2:3]
ends are exclusive
"""



"""
8. df.info(): this function provide us the details of the dataframe like in sql we get about table
it include the no. of columns, details about the columns in tha dataframe
"""

# print(df.info())

"""
We can modify the datatype of the columns also using the function:
- df[column name].astype(datatype)
now it return the new dataframe, so we need to 

df[col] = df[col].astype(datatype)
"""


"""
Sorting---------------------------------------------------------------------


Dataframes can be sort using the function
-df.sort_values(by = colname, ascending =T/F)
"""

# print(df.sort_values(by="Total intl minutes", ascending=True).head()) # as we can see here the dataset is sorted based on col = Total intl minutes. We used .head() so that we only fetch the first five rows

"""
in above we sort based on one col, we can sort the dataset based on multiple column
- but the 2nd colunm sort based on 1st, like first column sort first. then the second
- for each column we have to pas the ascending for each also
"""

# print(df.sort_values(by=["Churn", "Total intl minutes"], ascending=[False,True]).head())




"""
Indexing and Retrieving Data---------------------------------------------------------------------------

In pandas, we can access the column by simply passing the column name in the dataframe:
-df["col name"]

Boolean indexing with one column is also very convenient. The syntax is 
------------------------ df[P(df['Name'])]
where P is some logical condition that is checked for each element of the Name column. 
The result of such indexing is the DataFrame consisting only of the rows that satisfy the P condition on the Name column.
ex:
"""
# print(df.select_dtypes(np.number)[df["Churn"] == 1])
"""
df.select_dtypes: select only columns whose data type is same as the passed one.
np.number : include number 
df["Churn] == 1: we passed a condition in the dataframe, it will filter the table rows, and show only those who's Churn value is 1


now we can do more things on the dataframe
"""
# print(df.select_dtypes(np.number)[df["Churn"] == 1].mean()) # Mean of the each column, include rows only where the Churn value is 1

"""
Q: How much time (on average) do churned users spend on the phone during daytime?
"""
# print(df[df["Churn"] == 0]["Total day minutes"].mean())  # ans

"""
Q: What is the maximum length of international calls among loyal users (Churn == 0) who do not have an international plan?
"""

# print(df.info())
# print(df[(df["Churn"] == 0)&(df["International plan"] == "No")]["Total intl minutes"].max())


"""
In dataframes indexing take place by two ways:
1. loc[rown:rowm, Column1:colums2] : By name , By columns
    rown: nth row from where fetching begins
    rowm: mth row till then fetching take parts.(inclusive)
    column1: From which column
    column2: Till which column

2. iloc[rown:rowm, coli:colj] : By number, by rows
    instead of passing the name of column, it allow us to use the index of the columns

3. It also allow -ve indexing:
for printng the last row/col

[-1:]
    
    """

# print(df.info())
# print(df.loc[5:20, "State": "Voice mail plan"])

# print(df.iloc[5:20, 0:4])

#both the above statements work same, but using different ways and terms

# print(df.iloc[-1:])  #printed last row, take all column by default
# print(df.iloc[:-1])    # all rows and column

# print(df)

"""
Applying function to cells, column, row

- here we can apply a function to each cell using the function:

apply()

- if we want to apply this function of each row, we pass a argument axis = 1


- we can apply the function on a particular column like
    df[colName].apply(function)

eX:
"""
# print(df.apply(np.max)) # we applied the function np.max() on each column, where it return the max value of each column
# print(df.apply(np.max, axis=1))

# print(df["State"].apply(lambda state:state[0] == "W"))  #this will return the bool table with true where condition pass, false where condition fails

# print(df[df["State"].apply(lambda state:state[0] == "W")])  # return the rows where condition satisfied

"""
Map: We can use the map function to map the column with the condition or type.
also using the map function we can change the values of the column by passing a dictionary in it
"""
d = {"Yes": True, "No": False}
df["International plan"] = df["International plan"].map(d)
# print(df.head())


"""
replace(): same feature of map can be achieved using replace function. We pass column name and the dictionary of replacing values as value.

df.replace({column name: dictionary})
"""
df = df.replace({"Voice mail plan": d})




"""
Grouping----------------------------------------------------------------------------------------------------------------------------

df.groupby(by=grouping_columns)[columns_to_show].function()

1. First, the groupby method divides the grouping_columns by their values. They become a new index in the resulting dataframe.

2. Then, columns of interest are selected (columns_to_show). If columns_to_show is not included, all non groupby clauses will be included.

3. Finally, one or several functions are applied to the obtained groups per selected columns.

"""

# colToShow = ["Total day charge", "Total eve charge", "Total night charge"]
# df1 = df.groupby(by=["Churn"])[colToShow].describe()    # bind describe function with all the columns that are in coltoshow
# print(df1)

# colToShow = ["Total day charge", "Total eve charge", "Total night charge"]
# df1 = df.groupby(by=["Churn"])[colToShow].aggregate("min")  # return the min for each col by grouping Churn column
# print(df1)          


# colToShow = ["Total day charge", "Total eve charge", "Total night charge"]
# df1 = df.groupby(by=["Churn"])[colToShow].aggregate(["mean", "count", "min"]) # return all function values for the columns
# print(df1)


"""
Summary Tables

Suppose we want to see how the observations in our dataset are distributed in the context of 
two variables : "Churn" and "International plan". To do so, we can build a contingency table using the crosstab method:


pd.crosstab(df[col1], df[col2])

- if we pass normalize = True: it will provide us the proportion value instead of count
- if we pass margin = True, It will add two columns and rows of Grand total for the table
"""

# print(pd.crosstab(df["Churn"], df["International plan"]))
# print(pd.crosstab(df["Churn"], df["International plan"], normalize=True))
# print(pd.crosstab(df["Churn"], df["Voice mail plan"]))

# We can see that most of the users are loyal and do not use additional services (International Plan/Voice mail).

"""
Pivot table

This will resemble pivot tables to those familiar with Excel. And, of course, pivot tables are implemented 
in Pandas: the pivot_table method takes the following parameters:

values : a list of variables to calculate statistics for,

index : a list of variables to group data by,

aggfunc : what statistics we need to calculate for groups, e.g. sum, mean, maximum, minimum or something else

df.pivot_table(values , index,aggfunc())
"""
# print(df.info())
# print(df.pivot_table(["Total day calls", "Total eve calls", "Total night calls"], "Area code").mean())

# print(df.groupby(by="Area code")[["Total day calls", "Total eve calls", "Total night calls"]].aggregate("mean"))


"""
Data Transformation----------------------------------------------------------------------------------------------------------------------

Inserting column in the dataframe.-----------------------------------------------

df.insert(loc = index, column = column_name, value = values)
index: place at which the column will be placed in the dataframe

ex: We are inserting a column in df, Total calls, for the values we will take the sum of (Total calls day, eve, night)
"""
# print(df.info())

# total_calls = (df["Total day calls"] + df["Total eve calls"] + df["Total night calls"]+ df["Total intl calls"])
# df.insert(loc = len(df.columns), column="Total Calls", value=total_calls)

# the same thing can be achieved without insert function, But above gives you flexibility of the column place
# df["Total mins"] = (df["Total day minutes"]+df["Total eve minutes"] + df["Total night minutes"] + df["Total intl minutes"])
# print(df.head())


"""
Deleting the column or row

To delete columns or rows, use the drop method, passing the required indexes and the axis parameter 
(1 if you delete columns, and nothing or 0 if you delete rows). 
The inplace argument tells whether to change the original DataFrame. With inplace=False, 
the drop method doesn’t change the existing DataFrame and returns a new one with dropped rows or columns. 
With inplace=True, it alters the DataFrame.
If the columns not exists in the data drame, it will throw an error


Syntax:
    df.drop([col names], axis = n, inplace = T/F)

"""

# df.drop(["Total Calls", "Total mins"], axis=1, inplace=True)
# print(df.head())

# print(df.info())

# print(df.groupby("Area code")[["Total day calls", "Total eve calls", "Total night calls" , "Total intl calls"]].sum())

"""
Group by but with more better output format
"""

# for val, subdf in df.groupby("Area code"):
#     total = subdf["Total day calls"].sum() + subdf["Total eve calls"].sum()+subdf["Total night calls"].sum()+subdf["Total intl calls"].sum()
#     print(f"For Area code {val} the total calls are {total}")


"""If we group by two columns for a resulting column:"""
# print(df.groupby(["Area code", "International plan"])["Total day calls"].sum())





