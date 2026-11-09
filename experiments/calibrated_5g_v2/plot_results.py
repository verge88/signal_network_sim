import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1]); d=json.loads((root/'summary.json').read_text())
methods=['PREVIOUS','SPLIT','PERSIST','CARRY']
labels=['Прежний\nKALL','Раздельные\nпороги','Пороги и\nподтверждение роста','Пороги и\nсохранение истории']
security=[d['episode_summary'][m]['security'] for m in methods]
extra=[d['episode_summary'][m]['operational']-d['episode_summary'][m]['security'] for m in methods]
fp=[d['metrics']['all'][m]['security_fp'] for m in methods]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(11,8),height_ratios=[2,1],layout='constrained')
x=np.arange(4)
axes[0].bar(x,security,color='#245f73',label='Тревога безопасности')
axes[0].bar(x,extra,bottom=security,color='#d5a14d',label='Только уведомление о недоступности')
for i,(a,b) in enumerate(zip(security,extra)):
    axes[0].text(i,a+b+2,f'{a} + {b}',ha='center')
axes[0].set_ylim(0,130); axes[0].set_ylabel('Эпизодов из 108'); axes[0].legend(loc='upper left',fontsize=9,ncol=2)
axes[1].bar(x,fp,color='#a76562'); axes[1].bar_label(axes[1].containers[0],padding=3)
axes[1].set_ylim(0,max(2,max(fp)*1.4)); axes[1].set_ylabel('Ложные тревоги\nна нормальных окнах')
for ax in axes:
    ax.set_xticks(x,labels); ax.set_axisbelow(True); ax.grid(axis='y',alpha=.18)
fig.suptitle('Проверка раздельной калибровки и временного подтверждения',fontsize=14)
fig.supxlabel('3 новых стенда · по 20 минут теста · пороги зафиксированы до теста\nУведомление о недоступности не подтверждает атаку.',fontsize=10)
for ext in ['png','svg','pdf']:
    fig.savefig(root/f'comparison.{ext}',dpi=200,bbox_inches='tight')
