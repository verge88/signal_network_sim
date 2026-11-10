"""Scientific plots of measured results; no interpolation or inferred trials."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1]); data=json.loads((root/'summary.json').read_text())
rows=[r for r in data['metrics'] if r['primary']]
rows.sort(key=lambda r:['rules','IF_SBI','IF_CONTEXT','CARRY'].index(r['method']))
names=[r['method'] for r in rows]; x=np.arange(len(rows))
fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
for ax,key,title in zip(axes[:2],['recall','fpr'],['Полнота по окнам атак, %','Ложные тревоги на нормальных окнах, %']):
    ax.bar(x,[r[key]*100 for r in rows],color=['#8c8c8c','#4c78a8','#59a14f','#e58b32'],alpha=.7)
    for i,r in enumerate(rows):
        values=[p[key]*100 for p in r['per_run']]
        ax.scatter(np.full(3,i)+[-.1,0,.1],values,c='black',s=17,zorder=3)
    ax.set_title(title,fontsize=10);ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[1].axhline(.1,c='red',ls='--',lw=1,label='Ориентир 0,1%');axes[1].legend(fontsize=8)
ax=axes[2];ax.bar(x,[r['detected_episodes'] for r in rows],color='#4c78a8')
for i,r in enumerate(rows):ax.text(i,r['detected_episodes']+2,str(r['detected_episodes'])+'/108',ha='center',fontsize=9)
ax.set_ylim(0,120);ax.set_title('Эпизоды с тревогой безопасности',fontsize=10);ax.set_xticks(x,names,rotation=25,ha='right')
fig.suptitle('Новые общие данные: 3 × 1 час, 32 логические NF\nТочки — отдельные прогоны; недоступность не засчитывается как компрометация',fontsize=11)
fig.supxlabel('Наличие тревоги в эпизоде не доказывает её причину: у CARRY 2 из 88 эпизодов — D с сопутствующим превышением нагрузки.',fontsize=8)
for suffix in ['png','svg','pdf']:fig.savefig(root/('comparison.'+suffix),dpi=180)
plt.close(fig)
