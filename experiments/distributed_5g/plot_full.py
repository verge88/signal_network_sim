"""Scientific paired comparison; dots are runs, not independent NF windows."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    report = json.loads((root/'summary.json').read_text())
    names = ['rules','IF_SBI','IF_CONTEXT','WITHOUT_MINING','FULL_HYBRID']
    labels = ['Правила','IF: трафик','IF: контекст','Гибрид\nбез IF','Полный\nгибрид']
    methods = {m['method']:m for m in report['metrics']}
    fig, axes = plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax, metric, title in zip(axes,['recall','fpr'],['Полнота по окнам, %','Ложные тревоги (FPR), %']):
        values = [methods[name][metric]*100 for name in names]
        ax.bar(range(len(names)),values,color=['#a0a8b0','#83b8d0','#488eaf','#86b69c','#287852'],width=.6)
        for i,name in enumerate(names):
            runs = np.asarray(methods[name]['per_run_'+metric])*100
            ax.scatter(i+np.array([-.12,0,.12]),runs,s=24,color='#222222',zorder=3)
            ax.annotate(f'{values[i]:.2f}',(i,max(values[i],float(runs.max()))),xytext=(0,5),
                        textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks(range(len(names)),labels,fontsize=9)
        ax.set_title(title,fontsize=12)
        ax.set_ylim(bottom=0,top=max(1,max(ax.get_ylim())*1.15))
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        if metric=='fpr':
            ax.axhline(.1,color='#b53c38',ls='--',lw=1,label='Ориентир 0,1%')
            ax.legend(fontsize=8)
    fig.suptitle('Полный эксперимент 5G: одинаковые данные для всех вариантов',fontsize=13)
    fig.supxlabel('Столбцы — объединённая оценка; точки — три отдельных прогона. Это не доверительные интервалы.',fontsize=9)
    for extension in ('png','pdf','svg'):
        fig.savefig(root/f'ablation.{extension}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    plot(Path(sys.argv[1]))
