#######
# script for printing plots of different attack tests

import matplotlib.pyplot as plt
import numpy as np

# directory setup of test data
# base-dir: type of tests
# subdirs: target agent / attack type
base_dir = 'blackbox/'
subdirs = ['cw10/cwl2/', 'fgsm10/cwl2/']
# names of the log files without .txt
# log files need to have at least 100 episodes, first 100 are considered
files = ['p10', 'p20', 'p30', 'p40', 'p50', 'p60', 'p80']
colors = {'noattack': 'blue', 'cw10': 'yellowgreen', 'fgsm40': 'maroon', 'fgsm10': 'deeppink', 'fgsm20': 'orange'}

overall_rew = []
overall_perc = []

# x values in plot, attack probabilites in percent
x_labels = [10, 20, 30, 40, 50, 60, 80]

# for all target models / attack types
for sub_index, sub in enumerate(subdirs):

    all_rew = []
    all_percentages = []

    # for all attack probabilities that were computed
    for file_index, f in enumerate(files):

        # filename of logfile from base-directory
        filename = base_dir + sub + f + '.txt'

        # parsing through logfile
        # sum up rewards and success percentage
        # reads first 100 entries
        with open(filename, 'r') as fp:
            rew = 0
            percentage = 0
            counter = 0
            for line_index, line in enumerate(fp):
                tokens = line.strip().split(':')
                if len(tokens) > 1:
                    if tokens[0] == 'Reward':
                        rew += float(tokens[1])
                    if tokens[0] == 'Percentage of successful attacks':
                        percentage += float(tokens[1])
                        counter += 1

                if counter == 100:
                    break

            rew /= 100
            percentage /= 100
            all_rew.append(rew)
            all_percentages.append(percentage)

    print("subdir: ", sub)
    print("all percentages: ", all_percentages)
    print("avg return: ", all_rew)
    print("avg percentage: ", np.mean(all_percentages))

    overall_perc.append(all_percentages)
    overall_rew.append(all_rew)

# actual plotting
plt.title('fgsm blackbox attacks')
plt.xlabel('attack probability (%)')
plt.ylabel('average return')
plt.plot(x_labels, [20 for x in range(0, len(x_labels))], color='black', label='')
plt.plot(x_labels, overall_rew[0], color=colors['cw10'], label='cw10')
plt.plot(x_labels, overall_rew[1], color=colors['fgsm10'], label='fgsm10')

plt.legend()
plt.savefig("blackbox-fgsm.png")
plt.show()
