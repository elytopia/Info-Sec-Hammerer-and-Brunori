# script to plot learning curves

import matplotlib.pyplot as plt
import numpy as np

# directory containing the log files
base_dir = 'learning_progress/'
# name of the log files for the different models
files = ['no-attack', 'fgsm10', 'fgsm40', 'cwl2-p10', 'cwl2-p20']
colors = ['blue', 'deeppink', 'maroon', 'yellowgreen', 'cornflowerblue']

all_iters = []
all_rew = []

# for every log file
for count, f in enumerate(files):
    n_iters = []
    rew = []
    with open(base_dir + f + '.txt', 'r') as fp:

        # line parsing
        # write out iterations - reward pairs
        for index, line in enumerate(fp):
            tokens = line.replace('|', '').strip().split()
            if len(tokens) > 1:
                if tokens[0] == 'iters':
                    n_iters.append(tokens[1])
                if tokens[0] == 'reward':
                    rew.append(tokens[4])

    all_iters.append(n_iters)
    all_rew.append(rew)

print("parsed documents")

# actual plotting
fig, ax = plt.subplots()
for iter_id, x in enumerate(all_iters):

    y = all_rew[iter_id]
    y_arr = np.zeros(len(y))
    for y_index, y_val in enumerate(y):
        y_arr[y_index] = y_val

    x_arr = np.zeros(len(x))
    for x_index, x_val in enumerate(x):
        x_arr[x_index] = x_val

    ax.plot(x_arr, y_arr, color=colors[iter_id], label=files[iter_id])

plt.xlabel('number of iterations')
plt.ylabel('average return of the last 100 episodes')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(base_dir + 'overview.png')


print("computed plots")
print("done")
