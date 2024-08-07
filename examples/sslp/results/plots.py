import pandas as pd
import matplotlib.pyplot as plt


instance = 'sslp_5_25_50'
rho = 'rho1'
l1 = pd.read_csv(f'results/{instance}/l1-{rho}/PHHub/bounds.csv')

l2 = pd.read_csv(f'results/{instance}/l2-{rho}/PHHub/bounds.csv')

l1_bounds = l1['hub lower bound']
l2_bounds = l2['hub lower bound']

start = int(0 * len(l1_bounds))

plt.plot(l1_bounds[start:], label='l1')
plt.plot(l2_bounds[start:], label='l2')
plt.legend()
plt.title(f'Lagrangian lower bounds, {instance}')
plt.savefig(f'results/{instance}/l1-l2-{rho}.png')