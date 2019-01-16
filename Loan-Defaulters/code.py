# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df = pd.read_csv(path)
df_fico = df[(df['fico'] > 700)]
p_a = len(df_fico)/len(df)

df1 = df[df['purpose'] == 'debt_consolidation']
p_b = len(df1)/len(df)

df_new = df[(df['purpose'] == 'debt_consolidation') & (df['fico'] > 700)]
p_a_b = len(df_new)/len(df)

result = 0
if p_a_b == p_b:
 result = p_a_b
print(result)
# code ends here


# --------------
# code starts here
new_df = df[df['paid.back.loan'] == 'Yes']
prob_lp = len(new_df)/len(df)

df_credit_policy = df[df['credit.policy'] == 'Yes']
prob_cs = len(df_credit_policy)/len(df)

df_credit_loan = new_df[new_df['credit.policy'] == 'Yes']
prob_pd_cs = len(df_credit_loan)/len(new_df)

bayes = (prob_pd_cs * prob_lp)/prob_cs

print(bayes)
# code ends here


# --------------
# code starts here
import matplotlib.pyplot as plt

df1 = df[df['paid.back.loan'] == 'No']
plt.bar(df1['purpose'], height = 10)
plt.show()

# code ends here


# --------------
# code starts here
import matplotlib.pyplot as plt

inst_median = df['installment'].median()
inst_mean = df['installment'].mean()



# code ends here


