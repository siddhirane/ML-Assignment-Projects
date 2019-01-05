# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 



# code starts here

bank=pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)

numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)

# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'], axis = 1)
bank_mode = banks.mode()
banks.fillna(0, inplace = True)
print(banks.isnull().sum())
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks, index = ['Gender', 'Married', 'Self_Employed'], values = 'LoanAmount', aggfunc=np.mean)
print(avg_loan_amount)
# code ends here


# --------------
# code starts here
bank_se = banks[(banks['Self_Employed']== 'Yes') & (banks['Loan_Status']=='Y')]
loan_approved_se = bank_se.shape[0]
print(loan_approved_se)

bank_nse = banks[(banks['Self_Employed']== 'No') & (banks['Loan_Status']=='Y')]
loan_approved_nse = bank_nse.shape[0]
print(loan_approved_nse)

Loan_Status = 614

percentage_se = (loan_approved_se/Loan_Status) * 100
#print(percentage_se)
percentage_nse = (loan_approved_nse/Loan_Status) * 100
#print(percentage_nse)

# code ends here


# --------------
# code starts here
month = lambda x: int(x)/12

loan_term = banks['Loan_Amount_Term'].apply(month)

loan_df = loan_term.to_frame(name='My_loan_term')
loan_df.reset_index(drop=True, inplace=True)

loan_25 = loan_df[loan_df['My_loan_term'] >= 25]
big_loan_term = len(loan_25)

# code ends here


# --------------
# code ends here
loan_groupby = banks.groupby(['Loan_Status'])
loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]

mean_values = loan_groupby.mean()

# code ends here


