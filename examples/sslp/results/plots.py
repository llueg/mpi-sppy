import pandas as pd
import matplotlib.pyplot as plt


instance = 'sslp_5_25_50'
#instance = 'sslp_15_45_10'
rho = 'rho1'
l1 = pd.read_csv(f'results/{instance}/l1-{rho}/PHHub/bounds.csv')

l2 = pd.read_csv(f'results/{instance}/l2-{rho}/PHHub/bounds.csv')

sg = pd.read_csv(f'results/{instance}/l2-{rho}-sg/PHHub/bounds.csv')

l1_lb = l1['hub lower bound']
l2_lb = l2['hub lower bound']
sg_lb = sg['hub lower bound']
l1_ub = l1['hub upper bound']
l2_ub = l2['hub upper bound']
sg_ub = sg['hub upper bound']

start = int(0 * len(l1_lb))
start_ub = int(0.1 * len(l1_lb))
fig, ax = plt.subplots()

l1 = ax.plot(l1_lb[start:], label='l1')
ax.plot(l1_ub[start_ub:], linestyle='--', color=l1[0].get_color())
l2 = ax.plot(l2_lb[start:], label='l2')
ax.plot(l2_ub[start_ub:], linestyle='--', color=l2[0].get_color())
sg = ax.plot(sg_lb[start:], label='sg')
ax.plot(sg_ub[start_ub:], linestyle='--', color=sg[0].get_color())
l0 = ax.plot(l1_ub[start], linestyle='--', color='black', label='ub')
ax.legend()
ax.set_xlabel('Iteration')
ax.set_title(f'PH bounds, {instance}')
fig.savefig(f'results/{instance}/l1-l2-sg-{rho}.png')