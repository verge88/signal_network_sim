import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(sys.argv[1])
d=json.loads((root/'summary.json').read_text())
methods=['K','KR','KA','KT','KP','KALL']
labels=['Прежний\nK','+ Повторы','+ Потеря\nконтроля','+ Тренд','+ Тренд\nи разрешение','Все\nдополнения']
security=[d['episode_summary'][m]['security'] for m in methods]
additional=[d['episode_summary'][m]['operational']-d['episode_summary'][m]['security'] for m in methods]
fp=[d['metrics']['all'][m]['security_fp'] for m in methods]
incidents=[d['metrics']['all'][m]['availability_normal_episodes'] for m in methods]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,1,figsize=(11,10),height_ratios=[2,1,1],layout='constrained')
x=np.arange(len(methods))
axes[0].bar(x,security,color='#245f73',label='Есть тревога безопасности')
axes[0].bar(x,additional,bottom=security,color='#d5a14d',label='Только уведомление о потере контроля')
for i,(a,b) in enumerate(zip(security,additional)):
    axes[0].text(i,a+b+2,f'{a} + {b}',ha='center')
axes[0].set_ylim(0,125)
axes[0].set_ylabel('Эпизодов из 108')
axes[0].legend(loc='upper left',ncol=2,fontsize=9)
axes[1].bar(x,fp,color='#a76562')
axes[1].bar_label(axes[1].containers[0],padding=3)
axes[1].set_ylim(0,max(2,max(fp)*1.4))
axes[1].set_ylabel('Ложные тревоги\nбезопасности, окна')
axes[2].bar(x,incidents,color='#d5a14d')
axes[2].bar_label(axes[2].containers[0],padding=3)
axes[2].set_ylim(0,max(2,max(incidents)*1.4))
axes[2].set_ylabel('Отрезки потери\nконтроля на норме')
for ax in axes:
    ax.set_xticks(x,labels)
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Новая проверка повторов, доступности и изменения нагрузки',fontsize=15)
fig.supxlabel('3 независимых стенда · 20 минут теста на каждом\nУведомление о недоступности контроля не является подтверждением атаки.',fontsize=10)
for ext in ['png','svg','pdf']:
    fig.savefig(root/f'comparison.{ext}',dpi=200,bbox_inches='tight')
