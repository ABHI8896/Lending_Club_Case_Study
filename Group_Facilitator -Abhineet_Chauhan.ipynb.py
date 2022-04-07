#!/usr/bin/env python
# coding: utf-8

# ## LENDING CLUB CASE STUDY.
# 
# #### _First Importing the important libraries._

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Python Libraries: 


# #### _Now Importing the csv file named 'loan' from my local disk and naming the loaded dataframe as 'df'_

# In[2]:


df = pd.read_csv(r'C:\Users\ABHI\Downloads\Lending Club case study\loan\loan.csv')


# In[3]:


df.head()

# Dataframe Loaded in the notebook, Now Printing top 5 rows


# #### _Printing shape, info and describe command for DataFrame for better understanding_

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# #### _Finding missing values:_

# In[7]:


df.isnull().sum()

# There are very large number of missing values in the dataframe


# #### _Dropping the missing columns which has missing values more than 30 %, dropping the columns not required & defining New Dataframe (df1)_

# In[8]:


df1 = df.drop(['tot_hi_cred_lim','total_bal_ex_mort', 'total_bc_limit','url','total_il_high_credit_limit','num_tl_op_past_12m','revol_util','chargeoff_within_12_mths','collections_12_mths_ex_med','max_bal_bc','all_util','num_tl_90g_dpd_24m','pub_rec_bankruptcies','num_tl_120dpd_2m','num_tl_30dpd','pct_tl_nvr_dlq','percent_bc_gt_75','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','mths_since_recent_inq','mths_since_recent_revol_delinq','num_accts_ever_120_pd','num_actv_bc_tl','mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op','bc_open_to_buy','bc_util','acc_open_past_24mths','total_cu_tl','inq_last_12m','avg_cur_bal','total_rev_hi_lim','inq_fi','open_rv_12m','open_rv_24m','total_bal_il','il_util','open_il_24m','mths_since_rcnt_il','open_il_6m','open_il_12m','tot_cur_bal','open_acc_6m','tot_coll_amt','annual_inc_joint','next_pymnt_d','dti_joint','verification_status_joint','mths_since_last_delinq','mths_since_last_record','mths_since_last_major_derog','desc','tax_liens','title','emp_title','last_credit_pull_d'],axis=1)


# ***Checking the missing values again***

# In[9]:


df1.emp_length.isnull().sum()

# Here the column 'emp_lenght' has 1075 missing values
# So we will impute the missing value with 0.


# ***Imputing missing(NAN) values in the column of a dataframe which would be essential for our analysis***

# In[10]:


df1[~df1.emp_length.isnull()]


# In[11]:


df1.emp_length.fillna('0',inplace=True)
df1['emp_length']=df1.emp_length.str.extract('(\d+)')
df1['emp_length'].astype(int)


# In[12]:


df1.isnull().sum()

# we have successfully imputed the missing values in emp_length column 
# Now we will exclude column 'last_pymnt_d'


# In[13]:


df1 = df1[~df1.last_pymnt_d.isnull()]


# In[14]:


df1.last_pymnt_d.isnull().sum()


# ***We have successfully cleaning the data & handled missing values. Now performing sainity check of DataFrame***

# In[15]:


df1.isnull().sum()


# #### _Now knowing the data types*_

# In[16]:


df1.dtypes


# #### _Changing the data type & removing unnecessary symbols for better understanding_ 

# In[17]:


df1['int_rate'] = df1['int_rate'].replace('%', '', regex = True) 


# In[18]:


df1['int_rate'] = df1['int_rate'].astype(float)


# In[19]:


df1['term'] = df1['term'].replace('months', '', regex = True) 


# In[20]:


df1['term']=df1['term'].astype(float)


# ##### _Now, as we have imputed missing values, cleaned the data & changed the datatype of dataframe, Now we will start EDA_

# In[21]:


# Here we will go first with the dti (Debt-to-income Ratio), it could be the indicator of a person, how much net income he is left with each month.
 
df1.dti.value_counts()


# We can observe that the Person having lower 'dti' indicated good balance between debt and income. 
# The lower the percentage, the better the chance he will be a fully paid customer.


# In[22]:


sns.displot(df1.dti,color='cyan', kind="hist",aspect=1, bins=20)
plt.show()


# #### _Now Exploring more about dti & finding correlation of dti with some columns_

# In[23]:


df2 = df[["dti", "loan_amnt", 'installment','annual_inc','revol_bal']]


# In[24]:


# Now exploring the relationship between 'dti' & 'Loan amount'

plt.figure(figsize=(10,9))
sns.heatmap(df2.corr(),annot=True)

# Clearly, Acc. to the data 'dti' doesn't show any strong relationship with any of the above columns.
# so, we will move to further analysis.


# In[25]:


# Now we will look at the loan status:

df1.loan_status.value_counts()



# we can see that here out of 39717 loans 32950 have been fully paid only & 5556 are charged off


# In[26]:


# Now we calculate the percentage:

df1.loan_status.value_counts()*100/len(df1)

# we can see that the 83% of the loans are fully paid and aprrox 14% are charged off.
#  Now we will look at the parameters


# In[27]:


# Now we will Plot Histogram:


plt.figure(figsize=(7,6))
plt.hist(df1.loan_status, width =0.4, align='mid',color='skyblue',edgecolor='green')
plt.title('Loan Status',fontdict={'fontsize':18,'fontweight':5,'color':'blue'})
plt.xlabel('Loan Status',fontdict={'fontsize':14,'color':'green'})
plt.ylabel('Total Number of loans applications',fontdict={'fontsize':14,'color':'green'})
plt.xticks((0.2,1.2,2),['Fully Paid','Charged Off','Current'])
plt.show()


# This shows the amount of loans that were 'Charged Off'
# Current Loans are very less, approx 2.8% of the total loas applications.


# In[28]:


df1.purpose.value_counts()

# Most of the loans Taken were due to debt consolidation.
# The 2nd most purpose of loan is for credit card repayment.


# #### The below bar chart also shows that the most purpose of loan is for 'debt consolidation'

# In[29]:


# for better understanding we will plot countplot for purpose & loan status.


plt.figure(figsize=(10,9))
sns.countplot(y="purpose", data=df1,hue='loan_status')
plt.xlabel('\nNumber of loans applications',fontdict={'fontsize':15,'color':'green'})
plt.ylabel('\nPurpose of Loan\n',fontdict={'fontsize':15,'color':'green'})
plt.show()


# It is clear from the below chart that the people who take loan for debt consolidation are most likely to be 'charged off'.
# Also, the loan taken for Credit card repayment & small business have high risk of being 'Charged off'.


# #### _Now, we will check the column Home Ownership column_

# In[30]:


df1.home_ownership.value_counts()

# This shows that most of the loan applicants have rented house or property, to live.


# In[31]:


# Now we will try to find relationship between 'home_ownership' & 'Loan status'


plt.figure(figsize=(12,9))
sns.countplot(y="home_ownership", data=df1,hue='loan_status')
plt.xlabel('Loan Status\n',fontdict={'fontsize':12,'color':'black'})
plt.ylabel('Home Ownership',fontdict={'fontsize':14,'color':'black'})
plt.show()


# It is clear that applicant which has Rented ownership and Mortgage home are have high probablity of being 'Charged off'.
# The applicants who has 'Own Home Ownership' are less likely to get 'Charged Off'


# #### _Univariate Analysis on Interest Rate_ 

# In[32]:


df1.int_rate.describe()


# This Shows that average Interest Rate' is 12%.


# In[33]:


plt.figure(figsize=(7,5))
sns.boxplot(df1.int_rate)
plt.title('Box Plot-Interest Rate',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nInterest Rate\n',fontdict={'fontsize':14,'color':'black'})
plt.show()


# It shows that most of the 'Interest Rate' for the applicant is between 10% to 15% (approx.)


# #### _Now Exploring Interest Rate with Loan Status_

# In[34]:


plt.figure(figsize=(12,9))
sns.boxplot(df1.loan_status, df1.int_rate)
plt.title('Box Plot-Interest Rate\n',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nLoan Status\n',fontdict={'fontsize':14,'color':'black'})
plt.ylabel('\nInterest Rate\n',fontdict={'fontsize':14,'color':'black'})
plt.show()



# It shows that the most of the 'Charged Off' case have higher 'Interest Rate'.
# Whereas, 'Fully Paid' cases has lower Interest rate than.


# #### _Now, Exploring Loan Status with  Term_

# In[35]:


sns.barplot(x='loan_status', y='term', data=df1)
plt.title('Box Plot-Term Vs Loan Status\n',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nLoan Status\n',fontdict={'fontsize':14,'color':'black'})
plt.ylabel('\nTerm\n',fontdict={'fontsize':14,'color':'black'})
plt.show()



# This concludes that the Charged Off cases have higher Term.


# #### _Exploring Loan Amount_

# In[36]:


df1.loan_amnt.value_counts()*100/len(df1)

# This shows that 7% of the people took the loan amount of Rs.10000/-
# Whereas,6% of people took loan of Rs.12000/-


# In[37]:


df1.loan_amnt.describe()

# The average loan_amount is Rs.11224.65/- out of total cases.
# The maximum loan amount is Rs.35000/-


# In[38]:


# Loan Amount Vs Term Vs Loan Status:


plt.figure(figsize=(10,9))
sns.boxplot(y = 'loan_amnt', x = 'term', hue = 'loan_status', data = df1)
plt.title('Box-Plot Loan Amount Vs Term Vs Loan Status\n',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nTerm\n',fontdict={'fontsize':14,'color':'black'})
plt.ylabel('\nLoan Amount\n',fontdict={'fontsize':14,'color':'black'})
plt.legend(loc='upper right')
plt.show()


# This Shows that the loan taken for the period of 60 months and for loan amount of between Rs.10000 to Rs.20000
#these loans are more likely to be 'Charged off' than the same loan case for 36 months.


# In[39]:


# Exploring Interest rate with Term & Loan Status:

plt.figure(figsize=(10,9))
sns.boxplot(y = 'int_rate', x = 'term', hue = 'loan_status', data = df1)
plt.title('Box-Plot Interest Rate Vs Term Vs Loan Status\n',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nTerm\n',fontdict={'fontsize':14,'color':'black'})
plt.ylabel('\nLoan Interest Rate\n',fontdict={'fontsize':14,'color':'black'})
plt.legend(loc='upper right')
plt.show()


# This Shows that the Cases that were 'Charged Off' have Higher Interest Rate with longer period of time.
# The Boxplot shows 60 months loan term have higher Interest Rate.


# In[40]:


# Now exploring Loan Amount, funded amount & Funded Amount Invested.


plt.figure(figsize=(15,13))
plt.subplot(2, 3, 1)
sns.distplot(df1['loan_amnt'])
plt.title('Loan Amount - Distribution Plot\n',fontsize=14,color='y')
plt.xlabel('\nLoan Amount\n',fontsize=14,color='y')

plt.subplot(2, 3, 2)
sns.distplot(df1['funded_amnt'])
plt.title('Funded Amount - Distribution Plot\n',fontsize=14,color='y')
plt.xlabel('\nFunded Amount\n',fontsize=14,color='y')

plt.subplot(2, 3, 3)
sns.distplot(df1['funded_amnt_inv'])
plt.title('Funded Amount Inv. - Distribution Plot\n',fontsize=14,color='y')
plt.xlabel('\nFunded Amount Inv.',fontsize=14,color='y')
plt.show()

 
# This shows that the 'Distribution of amounts' for all three plots are similar.


# #### _Now we will look at the Annual Incomes of the applicants._

# In[41]:


plt.figure(figsize=(15,8))

x = np.linspace(0, 1e-9)
y = 1e3*np.sin(2*np.pi*x/1e-9)

plt.subplot(2, 2, 1)
sns.distplot(df1['annual_inc'])
plt.title('Annual Income - Distribution Plot',fontsize=14,color='r')
plt.xlabel('Annual Income',fontsize=14,color='y')

plt.subplot(2, 2, 2)
plt.title('Annual Income - Box Plot')
sns.boxplot(y=df1['annual_inc'])
plt.title('Annual Income - Box Plot',fontsize=16,color='w')
plt.ylabel('Annual Income',fontsize=14,color='w')
plt.show()


# #### _Now Looking at the Employment length Vs Loan Status_

# In[42]:


df1.emp_length.astype(int)


# In[43]:


df1.emp_length.value_counts()

# Most of the applicants are employed for more than 10 year.


# In[44]:


# Now, for employment length Vs Loan Status:

plt.figure(figsize=(15,8))
sns.countplot(x="loan_status", data=df1,hue='emp_length')
plt.title('Employment Length vs Loan Status',fontsize=14,color='black')
plt.xlabel('Loan_Status',fontsize=14,color='y')
plt.show()

# ItShows that the applicants with less than one year of employment length or no job have higher chance of getting'Charged off'


# #### _Now, Looking at the Grade in dataset_ 

# In[47]:


df1.grade.value_counts()


# In[83]:


# Analyzing Grade

plt.figure(figsize=(15,8))
plt.hist(df1.grade,bins=40,align='mid',color='skyblue',edgecolor='green')
plt.title('Grade',fontsize=14,color='black')
plt.ylabel('Number of Application',fontsize=14,color='black')
plt.xlabel('Grade',fontsize=14,color='black')
plt.show()


# Most of the applicants are B graded, followed by A grade.


# In[91]:


# Interest Rate Vs Grade

plt.figure(figsize=(10,9))
sns.boxplot(y = 'int_rate', x = 'grade', hue = 'loan_status', data = df1)
plt.title('Box-Plot Interest Rate Vs Grade Vs Loan Status\n',fontdict={'fontsize':15,'color':'black'})
plt.xlabel('\nGrade\n',fontdict={'fontsize':14,'color':'black'})
plt.ylabel('\nLoan Interest Rate\n',fontdict={'fontsize':14,'color':'black'})
plt.legend(loc='upper right')
plt.show()


#  The Lower grades are getting loans for higher interest rates because 'E','F','G' are more likely to 'Charged Off'


# In[58]:


plt.figure(figsize=(15,8))
sns.countplot(x="sub_grade", data=df1,hue='loan_status')
plt.title('subgrade Vs Loan Status',fontsize=14,color='black')
plt.xlabel('Loan_Status',fontsize=14,color='y')
plt.show()


# In[ ]:




