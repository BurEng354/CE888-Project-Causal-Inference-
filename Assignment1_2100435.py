import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot
import sklearn
from sklearn.model_selection import train_test_split

##########################################################
##Loading the Data
df_ihdp = pd.read_csv('https://raw.githubusercontent.com/dmachlanski/CE888_2022/main/project/data/ihdp.csv')
df_jobs = pd.read_csv('https://raw.githubusercontent.com/dmachlanski/CE888_2022/main/project/data/jobs.csv')

##Checking Datatypes of Values in the Columns
df_ihdp.info()
df_jobs.info()
print('Apparrently, all values in the Dataset are of type float64')

##Checking Summary Statistics
summary_stats_df_ihdp = df_ihdp.describe()
summary_stats_df_jobs = df_jobs.describe()

##Checking for Missing Values
##Note: First .sum() provides a series with information on missing values in each column
n_missing_val_ihdp = df_ihdp.isnull().sum().sum() 
n_missing_val_jobs = df_jobs.isnull().sum().sum()

if (n_missing_val_ihdp == 0 and n_missing_val_jobs == 0): 
    print("No Missing Values were detected in IHDP and JOBS respectively. Hence, Data Cleaning with regards to Missing Data is not required.")

#############################################################
##Dealing with Data Exploration of IHDP and JOBS individually
'''
For IHDP: Description of Varaibles:::::::::::::::::::::::::::

x1-x6 (Background Variables)-> Measurements about the child and Numerical Data regarding the Mother(e.g., child-birth weight, head
circumference, weeks born preterm, birth order, neonatal health index) 

x7-x25 (Categorical Background Variables)-> Mother's details at the time of Delivery (e.g., age, marital status, educational attainment, emplolyment status) and 
her behaviours during the pregnancy (e.g., whether she smoked cigarettes or not, consumed alcohol, had any drug intake...)
    
t (Categorical Treatment Variable)-> Support in the form of high-quality childcare and home visits (1 for provided; 0 otherwise)

yf (Outcome Variable showing Factual Outcomes)-> child's cognitive test score 

ycf-> counterfactual outcomes

ite-> True Individual Treatment Effect

'''


##For IHDP: Checking for Outliers  using Variables Definitions and Histograms.
##Also, menitioning useful Insights and deciding how to further Preprocess the Data
df_ihdp.hist(figsize = (20, 30), bins = 50)

   
'''        
For JOBS = Description of Variables:::::::::::::::::::::::::::
    
x1-x17-> Showing often-used numerical Background Characteritics -- (Background Variables)
t-> Reception of Job Training under NSWP (1 for Training Received; O for no Training Received) -- (Treatment Varaible)
y-> Employment Status (1 for Employed; 0 otherwise) -- (Outcome Variable)
e-> Source of Sample (1 for Experimental Data; 0 for Observational Data) -- It will be dropped to reduce Overfitting (as all the data is to be used according to the assignment document)
    but we could have deleted rows with Observational Data (for which e = 0) and dropped column e afterwards 
    to have a more reliable Dataset and better-performing CI Model
    
    Possible definitions of x1-x17 from Tables in References [5] and [4] in the Assignmnet Document: 
    x1-x2 (in all Tables Age and Education/School were at the Start hence x1 and x2 probably refer to them)
    Age, Age of Participant 
    Education/School, number of school years; 

    ***NOTE: Both x1 and x2 were presumably Scaled as they would have reasonable positive values otherwise.
    They seem to have been Scaled using a Scalar robust to Outliers as Outliers are not so visually obvious in their Histogram shown below. 
    A second Scaling to remove Outliers using Robust Scalars was considered but sidelined.
    https://scikit-learn.org/stable/modules/preprocessing.html
    https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py

    x3-x6:
    Black, 1 if black, 0 otherwise; 
    Hisp, 1 if Hispanic, 0 otherwise; 
    No degree, 1 if participant had no school degrees, 0 otherwise; 
    Married, 1 if married, 0 otherwise; 
    x14, x15, x17:
    U74, 1 if unemployed in 1974, 0 otherwise; 
    U75, 1 if unemployedin 1975, 0 otherwise;
    ...
    
    Remamining Background Varaibles (were Scaled same as x1 and x2)
    RE74, real earnings(1982US$)in 1974; (1 – Inflation Rate) * Wages = real income (from Investopedia we see that real income can be negative)
    RE75, real earnings(1982US$)in 1975;
    RE78, real earnings...

'''

##For JOBS: Checking for Outliers  using Variables Definitions and Histograms.
##Also, menitioning useful Insights and deciding how to further Preprocess the Data
df_jobs_eRem = df_jobs.drop(columns = ['e'])
df_jobs_eRem.hist(figsize = (20, 30), bins = 50)


'''
For JOBS: Insights and Preprocessing Strategies:
    1) x3-x6, x14, x15, x17 are Categorical Varaibles (No Encoding required as they have Binary Values)
    The Histograms show that none of the Categoircal Varaibles 
    have any Outliers in their Columns (only 0 or 1 values are there )
    
    2) For certain variables like x1, x2, and the remaining Varaibles aside from those in 1):
    They will be Normalized/Scaled again using a scalar robust to Outliers to deal with possible Outliers after the first Scaling
    
    3) No Temporal Data is present that needs to be converted to Real Numbers using a Reference Time
    
    4) Some of the Variables (x7, x8) have some skeweness in their distribution. 
       The Skewness will be reduced further with the 2nd Scaling to reduce errors due to skewed Results
    
    5) As correlations do not imply causation, the correlations between 
    the variables were not used for generating insight as they did not have any 
    value given that the problem was one of Causal Inference. 
'''


##Test and Train Split of IHDP and JOBS (Part of Task 3)::::
##For IHDP:
x_ihdp = df_ihdp.iloc[:,:25]
t_ihdp = df_ihdp['t']
yf_ihdp = df_ihdp['yf']
ycf_ihdp = df_ihdp['ycf']
ite_ihdp = df_ihdp['ite']
x_train_ihdp, x_test_ihdp, t_train_ihdp, t_test_ihdp, yf_train_ihdp, yf_test_ihdp, ycf_train_ihdp, ycf_test_ihdp, ite_train_ihdp, ite_test_ihdp = train_test_split(x_ihdp, t_ihdp, yf_ihdp, ycf_ihdp, ite_ihdp, test_size=0.2)
'''
Note: ycf_train_ihdp, ite_train_ihdp will be discarded as they are not to be used during training
ite will be used during evaluation (yf and ycf will not be as they are contaminated by noise).
'''

##For JOBS:
x_jobs = df_jobs_eRem.iloc[:,:17]
t_jobs = df_jobs_eRem['t']
y_jobs = df_jobs_eRem['y']
x_train_jobs, x_test_jobs, t_train_jobs, t_test_jobs, y_train_jobs, y_test_jobs = train_test_split(x_jobs, t_jobs, y_jobs, test_size=0.2, stratify = y_jobs)

 
