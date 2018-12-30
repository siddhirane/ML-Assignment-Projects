# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)

data['Gender'].replace('-', 'Agender',inplace=True)
#Code starts here 
gender_count = data['Gender'].value_counts()
plt.bar(gender_count, height= 10)
plt.show()



# --------------
#Code starts here
alignment = data['Alignment'].value_counts()
plt.pie(alignment)
plt.title('Character Alignment')
plt.show()


# --------------
#Code starts here
sc_df = data[['Strength','Combat']]
sc_covariance = sc_df['Strength'].cov(sc_df['Combat'])
sc_strength = sc_df['Strength'].std()
sc_combat = sc_df['Combat'].std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)

ic_df = data[['Intelligence','Combat']]
ic_covariance = ic_df['Intelligence'].cov(ic_df['Combat'])
ic_intelligence = ic_df['Intelligence'].std()
ic_combat = ic_df['Combat'].std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)


# --------------
#Code starts here
total_high = data['Total'].quantile(0.99)
super_best = data[data['Total'] > total_high]
super_best_names = [super_best['Name']]
print(super_best_names)



# --------------
#Code starts here
fig = plt.figure()
ax_1 = fig.add_axes()
ax_2 = fig.add_axes()
ax_3 = fig.add_axes()


