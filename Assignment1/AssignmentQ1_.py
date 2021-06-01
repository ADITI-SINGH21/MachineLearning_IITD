import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statistics
df_1 = pd.read_excel('data_1.xlsx')
df_1.plot.scatter(x = 'x', y = 'y')
print('The scatter plot for data 1')
plt.show()
pd.DataFrame(df_1).plot.hist()
print('Histogram Plot for data 1')
plt.show()
sns.heatmap(pd.DataFrame(df_1))
print('Heat map for 1st data')
plt.show()
sns.boxplot(x=df_1.x)
print('Boxplot for 1st column')
plt.show()
sns.boxplot(x=df_1.y)
print('Boxplot for 2nd column')
plt.show()
df_3 = pd.read_excel('data_3.xlsx')
df_3.plot.scatter(x = 'x', y = 'y')
print('Scatter Plot for data 3')
plt.show()
pd.DataFrame(df_3).plot.hist()
print('Histogram Plot for Data3 column x')
plt.show()
sns.heatmap(pd.DataFrame(df_3))
print('Heat map for 2nd data')
plt.show()
sns.boxplot(x=df_3.x)
plt.title('Box Plot for x column 2nd data')
sns.boxplot(x=df_3.y)
plt.title('Box Plot for y column 2nd data')
print('The statistics for first data are')
print(df_1.describe())
print('The statistics for second data are')
print(df_3.describe())
print('Using standard deviation approach the outliers are')
z_score = np.abs(stats.zscore(df_3))
arrayx=[]
arrayy=[]
for i in range(len(df_3)):
    if z_score[:,0][i]>3:
        arrayx.append(np.array(df_3)[:,0][i])
    if z_score[:,1][i]>3:
        arrayx.append(np.array(df_3)[:,1][i])
print(arrayx)
print(arrayy)
print('Using MAD approach the outliers are')
def zm_score(df):
    Median = statistics.median(df)
    MAD = statistics.median(np.abs(df-Median))
    Z_modified = 0.6745*(df-Median)/MAD
    return Z_modified
ax=[]
ay=[]
z1 = zm_score(df_3.x)
z2 = zm_score(df_3.y)
for i in range(len(df_3)):
    if z1[i]>3:
        ax.append(np.array(df_3)[:,0][i])
    if z2[i]>3:
        ay.append(np.array(df_3)[:,1][i])
print(ax)
print(ay)
