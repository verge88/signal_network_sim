import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]); s=json.loads((root/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors={'early':'#89969c','late':'#245f73'}; names={'early':'Прежний порядок','late':'Перед решением'}
x=np.arange(3)
for shift,mode in [(-.18,'early'),(.18,'late')]:
    p=[v[mode+'_timing'] for v in s['pairs']]
    values=[100*v['normal_reports_aging_after_return']/max(1,v['normal_reports_fresh_at_return']) for v in p]
    bars=axes[0,0].bar(x+shift,values,.36,label=names[mode],color=colors[mode]); axes[0,0].bar_label(bars,fmt='%.2f',padding=3)
    bars=axes[0,1].bar(x+shift,[v['normal_report_age_p95_ms'] for v in p],.36,label=names[mode],color=colors[mode]); axes[0,1].bar_label(bars,fmt='%.2f',padding=3)
axes[0,0].axhline(.1,color='#b47428',linestyle='--',linewidth=1,label='Критерий 0,1%')
axes[0,0].set(title='Отчёты, устаревшие после получения',ylabel='% нормальных отчётов, свежих при возврате',xticks=x,xticklabels=['33101','33201','33301'],xlabel='Исходная модель (seed)')
axes[0,0].legend(fontsize=8)
axes[0,1].axhline(250,color='#b47428',linestyle='--',linewidth=1,label='Предел 250 мс')
axes[0,1].set(title='Возраст нормального отчёта при решении',ylabel='p95, мс',xticks=x,xticklabels=['33101','33201','33301'],xlabel='Исходная модель (seed)')
axes[0,1].legend(fontsize=8)
x=np.arange(4)
for shift,mode in [(-.18,'early'),(.18,'late')]:
    scenarios=s['by_mode'][mode]['episodes']['CARRY']['scenarios']
    vals=[scenarios[k]['security'] for k in ['C','F','R']]+[scenarios['D']['availability']]
    bars=axes[1,0].bar(x+shift,vals,.36,label=names[mode],color=colors[mode]);axes[1,0].bar_label(bars,padding=3)
axes[1,0].set(title='Контроль конфигурации и протокола',ylabel='Эпизодов из 12',xticks=x,xticklabels=['C: изменение','F: подделка','R: повтор','D: недоступность'],ylim=(0,16))
axes[1,0].legend(fontsize=8)
for i,mode in enumerate(['early','late']):
    m=s['by_mode'][mode]['metrics']['slot-4']['CARRY']; n=m['normal_windows']
    unknown=100*m['unknown_normal']/n; unavailable=100*m['availability_normal_windows']/n
    axes[1,1].bar(i,unknown,color='#89969c',label='Unknown' if i==0 else None)
    axes[1,1].bar(i,unavailable,bottom=unknown,color='#d5a14d',label='Недоступность' if i==0 else None)
    axes[1,1].text(i,unknown+unavailable+.3,f'{unknown+unavailable:.2f}%',ha='center')
axes[1,1].set(title='Нормальные окна без определённого контроля',ylabel='% нормальных окон slot-4',xticks=[0,1],xticklabels=[names['early'],names['late']])
axes[1,1].legend(fontsize=8)
for ax in axes.flat: ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18);ax.margins(y=.15)
fig.suptitle('Получение контрольного отчёта перед решением · 32 NF',fontsize=16)
fig.supxlabel('Три новые сопоставленные пары · прежние модели и предел свежести 250 мс\nРазные облачные runner; уведомление о недоступности не доказывает атаку.',fontsize=10)
for ext in ['png','svg','pdf']: fig.savefig(root/f'comparison.{ext}',dpi=200,bbox_inches='tight')
