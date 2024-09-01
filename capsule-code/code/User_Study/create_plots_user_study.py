import numpy as np
import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rc

sns.set(style='ticks', palette='colorblind')
sns.despine()


def main():
	# load excel data file
	df = pd.read_excel('data/user_study/Gesamtdatensatz_R_subset.xlsx')

	TC1 = df.loc[df['VB'] == 1][['VertrauenRegel1', 'VertrauenRegel2', 'VertrauenRegel3']].to_numpy()
	TC2 = df.loc[(df['VB'] == 2)][['VertrauenRegel1', 'VertrauenRegel2', 'VertrauenRegel3']].to_numpy()
	TC3 = df.loc[df['VB'] == 4][['VertrauenRegel1', 'VertrauenRegel2', 'VertrauenRegel3']].to_numpy()
	TiA_TC1 = df.loc[df['VB'] == 1][['TiAQ_Score']].to_numpy(dtype=float)
	TiA_TC2 = df.loc[df['VB'] == 2][['TiAQ_Score']].to_numpy(dtype=float)
	TiA_TC3 = df.loc[df['VB'] == 4][['TiAQ_Score']].to_numpy(dtype=float)

	# remove samples with missing values
	TC1 = TC1[~np.isnan(TC1).any(axis=1)]
	TC2 = TC2[~np.isnan(TC2).any(axis=1)]
	TC3 = TC3[~np.isnan(TC3).any(axis=1)]

	TiA_TC1 = TiA_TC1[~np.isnan(TiA_TC1).any(axis=1)]
	TiA_TC2 = TiA_TC2[~np.isnan(TiA_TC2).any(axis=1)]
	TiA_TC3 = TiA_TC3[~np.isnan(TiA_TC3).any(axis=1)];

	rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
	rc('text', usetex=False)

	plt.style.use('ggplot')

	axislabel_fontsize = 24
	ticklabel_fontsize = 22
	titlelabel_fontsize = 26

	# set grid layout
	fig = plt.figure(constrained_layout=False, figsize=(20, 8))
	gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.0, right=0.25,
							wspace=0.05)
	f_ax1 = fig.add_subplot(gs1[0, 0])

	gs2 = fig.add_gridspec(nrows=1, ncols=3, left=0.3, right=1,
							wspace=0.01)
	f_ax2 = fig.add_subplot(gs2[:, 0])
	f_ax3 = fig.add_subplot(gs2[:, 1])
	f_ax3.yaxis.set_ticks_position('none')
	plt.setp(f_ax3.get_yticklabels(), visible=False)
	f_ax4 = fig.add_subplot(gs2[:, 2])
	f_ax4.yaxis.set_ticks_position('none')
	plt.setp(f_ax4.get_yticklabels(), visible=False)

	# figure trust in ai
	data = [TiA_TC1, TiA_TC2, TiA_TC3]
	sns.boxplot(data=data, ax=f_ax1)
	f_ax1.set_ylim(0.5, 6)
	f_ax1.set_ylabel('TiA Score', size=axislabel_fontsize)
	f_ax1.set_yticks([1, 2, 3, 4, 5])
	f_ax1.set_xlabel('Test Conditions', size=axislabel_fontsize)
	f_ax1.set_xticklabels(['TC1', 'TC2', 'TC3'])
	f_ax1.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
	f_ax1.set_title('(a) Trust in AI', size=titlelabel_fontsize)

	# statistical annotation
	x1, x2, x3 = 0, 1, 2
	y, h, col = 5 + 0.4, 0.1, 'k'
	f_ax1.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax1.text((x1+x3)*.5, y+0.1, "**", ha='center', va='bottom', color=col, size=ticklabel_fontsize)
	y, h, col = 5 + 0.1, 0.1, 'k'
	f_ax1.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax1.text((x2+x3)*.5, y+0.1, "***", ha='center', va='bottom', color=col, size=ticklabel_fontsize)

	# figure trust rule learning tc1
	sns.boxplot(data=TC1, orient='v', ax=f_ax2, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0])
	f_ax2.set_ylim(0.5, 6)
	f_ax2.set_ylabel('Trust in Correct Rule Learning', size=axislabel_fontsize)
	f_ax2.yaxis.set_ticks_position('none')
	f_ax2.set_xticklabels(['50\%', '70\%', '100\%'])
	f_ax2.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
	f_ax2.set_title('(b) No Explanations\n(TC1)', size=titlelabel_fontsize)

	# statistical annotation
	x1, x2, x3 = 0, 1, 2
	y, h, col = 5 + 0.4, 0.1, 'k'
	f_ax2.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax2.text((x1+x3)*.5, y+0.1, "***", ha='center', va='bottom', color=col, size=ticklabel_fontsize)
	y, h, col = 5 + 0.1, 0.1, 'k'
	f_ax2.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax2.text((x2+x3)*.5, y+0.1, "***", ha='center', va='bottom', color=col, size=ticklabel_fontsize)

	# figure trust rule learning tc2
	sns.boxplot(data=TC2, orient='v', ax=f_ax3, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
	f_ax3.set_ylim(0.5, 6)
	f_ax3.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
	f_ax3.set_xticklabels(['50\%', '70\%', '100\%'])
	f_ax3.yaxis.set_ticks_position('none')
	f_ax3.set_title('(c) Correct Explanations \n(TC2)', size=titlelabel_fontsize)
	f_ax3.set_xlabel('Model Accuracy per Round', size=axislabel_fontsize)

	# statistical annotation
	x1, x2, x3 = 0, 1, 2
	y, h, col = 5 + 0.4, 0.1, 'k'
	f_ax3.plot([x1, x1, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax3.text((x1+x3)*.5, y+0.1, "***", ha='center', va='bottom', color=col, size=ticklabel_fontsize)
	y, h, col = 5 + 0.1, 0.1, 'k'
	f_ax3.plot([x2, x2, x3, x3], [y, y+h, y+h, y], lw=1.5, c=col)
	f_ax3.text((x2+x3)*.5, y+0.1, "***", ha='center', va='bottom', color=col, size=ticklabel_fontsize)

	# figure trust rule learning tc4
	sns.boxplot(data=TC3, orient='v', ax=f_ax4, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][2])
	f_ax4.set_ylim(0.5, 6)
	f_ax4.tick_params(axis='both', which='major', labelsize=ticklabel_fontsize)
	f_ax4.set_xticklabels(['50\%', '70\%', '100\%'])
	f_ax4.yaxis.set_ticks_position('none')
	f_ax4.set_title('(d) Incorrect Explanations \n(TC3)', size=titlelabel_fontsize);

	os.makedirs('results/user_study/', exist_ok=True)
	fig.savefig('results/user_study/user_study_boxplots.pdf', bbox_inches='tight')


if __name__ == "__main__":
	main()
