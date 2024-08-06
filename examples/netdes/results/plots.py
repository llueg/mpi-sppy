import pandas as pd
import matplotlib.pyplot as plt

l1 = pd.read_csv('network-10-20-L-01/l1-rho3e3-sg/PHHub/bounds.csv')

l2 = pd.read_csv('network-10-20-L-01/l2-rho3e3-sg/PHHub/bounds.csv')

l1_bounds = l1['hub lower bound']
l2_bounds = l2['hub lower bound']

plt.plot(l1_bounds, label='l1')
plt.plot(l2_bounds, label='l2')
plt.legend()
plt.savefig('network-10-20-L-01/l1-l2-rho3e3-sg.png')