# bar plots of average return and attack success probability
import matplotlib.pyplot as plt
import numpy as np

# base directory containing the log files
base_dir = 'blackbox/'
# subdirectories - target models
subdirs = ['cw10/', 'fgsm40/', 'noattack/']
# substitute models for the corresponding target models
# file names without .txt
files = [['fgsm10', 'fgsm20', 'fgsm40', 'no-attack'], ['cw10', 'fgsm10', 'no-attack'], ['cw10', 'fgsm10', 'fgsm40']]
colors = {'noattack': 'blue', 'cw10': 'yellowgreen', 'fgsm40': 'maroon', 'fgsm10': 'deeppink', 'fgsm20': 'orange'}

overall_rew = []
overall_perc = []

# for all target models
for sub_index, sub in enumerate(subdirs):

    all_rew = []
    all_percentages = []

    # for all substitute models
    for file_index, f in enumerate(files[sub_index]):

        # log file name from base-dir
        filename = base_dir + sub + f + '.txt'

        # parse log file
        # sum up reward and percentage
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

    overall_perc.append(all_percentages)
    overall_rew.append(all_rew)

    plt.bar(files[sub_index], all_percentages, align='center', alpha=0.5, color=colors[sub.strip('/')])
    plt.xticks(files[sub_index])
    plt.ylabel('percentage of successful attacks')
    plt.xlabel('model to compute attacks')
    plt.title(sub.strip('/'))

    plt.tight_layout()
    plt.savefig(base_dir + sub + sub.strip('/') + '-percentage.png')
    plt.close()

# set width of bar
barWidth = 0.25

# set height of bar
# first set of bars - attacks on noattack
bars1 = [overall_perc[2][0], overall_perc[2][1], overall_perc[2][1], 0]
# second bar - attacks on fgsm40
bars2 = [overall_perc[1][0], overall_perc[1][1], 0, overall_perc[1][2]]
# third bar - atacks on cwl2
bars3 = [0, overall_perc[0][0], overall_perc[0][1], overall_perc[0][2]]


# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Make the plot
plt.bar(r1, bars1, color=colors['noattack'], width=barWidth, edgecolor='white', label='noattack')
plt.bar(r2, bars2, color=colors['fgsm40'], width=barWidth, edgecolor='white', label='fgsm40')
plt.bar(r3, bars3, color=colors['cw10'], width=barWidth, edgecolor='white', label='cwl2')
# plt.bar(r4, bars4, color='red', width=barWidth, edgecolor='white', label='test')

# Add xticks on the middle of the group bars
plt.ylabel('percentage of successful attacks')
plt.xlabel('substitute model')
plt.xticks([r + barWidth for r in range(len(bars1))], ['cw10', 'fgsm10', 'fgsm40', 'no attack'])
plt.title("continous black box test time attacks (cwl2)")
# Create legend & Show graphic
plt.legend(title="target model", loc='lower right')
plt.tight_layout()
plt.savefig('blackbox-percentage.png')
plt.close()

bars1 = [overall_rew[2][0], overall_rew[2][1], overall_rew[2][1], 0]
# second bar - attacks on fgsm40
bars2 = [overall_rew[1][0], overall_rew[1][1], 0, overall_rew[1][2]]
# third bar - atacks on cwl2
bars3 = [0, overall_rew[0][0], overall_rew[0][1], overall_rew[0][2]]
plt.bar(r1, bars1, color=colors['noattack'], width=barWidth, edgecolor='white', label='noattack')
plt.bar(r2, bars2, color=colors['fgsm40'], width=barWidth, edgecolor='white', label='fgsm40')
plt.bar(r3, bars3, color=colors['cw10'], width=barWidth, edgecolor='white', label='cw10')

# Add xticks on the middle of the group bars
plt.ylabel('average return')
plt.xlabel('substitue model')
plt.xticks([r + barWidth for r in range(len(bars1))], ['cw10', 'fgsm10', 'fgsm40', 'no attack'])
plt.title("continous black box test time attacks (cwl2)")
# Create legend & Show graphic
plt.legend(title="target model", loc='upper right')
plt.tight_layout()
plt.savefig('blackbox-return.png')
