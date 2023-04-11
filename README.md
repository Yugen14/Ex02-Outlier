### EX -02 OUTLIER

# AIM
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
 
# EXPLANATION
   
 An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.  

# ALGORITHM

### STEP 1
Read the given Data.

### STEP 2
Get the information about the data.

### STEP 3
Detect the Outliers using IQR method and Z score.

### STEP 4
Remove the outliers.

### STEP 5
Plot the datas using Box Plot.

# PROGRAM

```
Developed by : yugendhar
Registration Number : 212220040184
```
```
import pandas as ps
import numpy as np
import seaborn as sns

df=ps.read_csv("bhp.csv")
df

df.head()
df.describe()
df.info()
df.isnull().sum()
df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

IQR=q3-q1 #INTERQUARTILE RANGE
ul =q3+0.5*IQR
ll =q1-1.5*IQR

df1=df[((df['price_per_sqft']<=l1)&(df['price_per_sqft']>u1))]
df1

df1.shape

sns.boxplot(x='price_per_sqft',data=df1)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

print(df2.shape)

sns.boxplot(x='price_per_sqft',data=df2)

df3=ps.read_csv('height_weight.csv')
df3

df3.head()
df3.info()
df3.describe()
df3.isnull().sum()
df3.shape

sns.boxplot(x='weight',data=df3)

q1=df3['weight'].quantile(0.25)
q3=df3['weight'].quantile(0.75)
print('First Quantile =',q1,'Second Quantile =',q3)

IQR=q3-q1
u1=q3+1.5*IQR
l1=q1-1.5*IQR

df4 =df3[((df3['height']>=l1)&(df3['height']<=u1))]
df4

df4.shape

sns.boxplot(x='height',data=df4)
```

# OUTPUT

### DATASET FOR BHP_CSV
![1](https://user-images.githubusercontent.com/129398164/229768714-121464ca-2e43-45cd-939b-6870503d03b9.png)

### DATASET HEAD(BHP)
![2](https://user-images.githubusercontent.com/129398164/229768744-1cd72670-43f3-4fa2-b148-a38ce6603b82.png)

### DATASET DESCRIBE(BHP)
![3](https://user-images.githubusercontent.com/129398164/229768765-932aeb53-bcfd-4cd4-a243-221b5fd5209f.png)

### DATASET INFO(BHP)
![4](https://user-images.githubusercontent.com/129398164/229768907-516a034e-a166-41ea-b285-4e95e5a3c8a6.png)

### DATASET NULL VALUES(BHP)
![5](https://user-images.githubusercontent.com/129398164/229768922-888f0c6b-3c56-4d45-a946-4fcda076c25d.png)

### DATASET SHAPE WITH OUTLIERS(BHP)
![6](https://user-images.githubusercontent.com/129398164/229769018-f880c2d4-135f-4a78-b73b-a5777e684e54.png)

### DATASET BOXPLOT WITH OUTLIERS(BHP)
![7](https://user-images.githubusercontent.com/129398164/229769031-bc257dd5-6af2-49ac-8a80-7d57d720c714.png)

### DATASET WITHOUT OUTLIERS(BHP)

![8](https://user-images.githubusercontent.com/129398164/229769041-51bb0e9e-d679-40e8-b260-8bb725c46210.png)

### DATASET SHAPE WITHOUT OUTLIERS(BHP)
![9](https://user-images.githubusercontent.com/129398164/229769062-e33a1dd3-5afa-4873-a0a2-cdf5e50f972c.png)

### DATASET BOXPLOT WITHOUT OUTLIERS(BHP)
![10](https://user-images.githubusercontent.com/129398164/229769078-ccfa02cb-b636-453a-a6ae-97f1c575566f.png)

### DATASET AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![11](https://user-images.githubusercontent.com/129398164/229769085-c4c33c28-b4ac-46da-8c4b-84982f4487a3.png)

### DATASET SHAPE AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![12](https://user-images.githubusercontent.com/129398164/229769094-bb466d77-efe2-4d44-8e2d-262bddd0ae49.png)

### DATASET BOXPLOT AFTER REMOVAL OF OUTLIERS USING Z-SCORE(BHP)
![13](https://user-images.githubusercontent.com/129398164/229769106-c4646105-39b8-451f-b7f8-c2fa575d9ded.png)

### DATASET FOR WEIGHT_HEIGHT_CSV
![14](https://user-images.githubusercontent.com/129398164/229769119-1f536714-9f51-4dcb-a2ed-0a6495be457b.png)

### DATASET HEAD(WEIGHT_HEIGHT)
![15](https://user-images.githubusercontent.com/129398164/229769186-d87d7391-f18a-45f5-8148-009487c2b9d4.png)

### DATASET INFO(WEIGHT_HEIGHT)
![16](https://user-images.githubusercontent.com/129398164/229769194-ffd03750-8f78-4b0e-971a-c2854610a513.png)

### DATASET DESCRIBE(WEIGHT_HEIGHT)
![17](https://user-images.githubusercontent.com/129398164/229769205-2dcd9a58-c970-472f-abf8-f4ef882e6a00.png)

### DATASET NULL VALUES(WEIGHT_HEIGHT)
![18](https://user-images.githubusercontent.com/129398164/229769231-d8862e34-9aac-44aa-9390-c7f41d072840.png)

### DATASET BOXPLOT WITH OUTLIERS(WEIGHT_HEIGHT)
![19](https://user-images.githubusercontent.com/129398164/229769245-6107f98e-98be-421a-a933-441d774b69be.png)

### DATASET AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)

![20](https://user-images.githubusercontent.com/129398164/229769259-18d39530-b90f-4dbf-a26c-f4e3ab5f3889.png)


### DATASET SHAPE(WEIGHT_HEIGHT)
![21](https://user-images.githubusercontent.com/129398164/229769268-27021992-8994-4cf7-adc8-a3d91cdf3a57.png)

### DATASET BOXPLOT AFTER REMOVING OUTLIERS USING IQR METHOD(WEIGHT_HEIGHT)
![22](https://user-images.githubusercontent.com/129398164/229769348-8cee0931-09f3-4cb4-97fa-fe3530fcfd9f.png)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
