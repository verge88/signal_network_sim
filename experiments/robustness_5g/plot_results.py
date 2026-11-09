import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(sys.argv[1]); d=json.loads((root/'summary.json').read_text()); overall=d['overall']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
methods=['PREVIOUS','SPLIT','PERSIST','CARRY']; x=np.arange(4)
security=[overall['episodes'][m]['security'] for m in methods]
extra=[overall['episodes'][m]['operational']-overall['episodes'][m]['security'] for m in methods]
axes[0,0].bar(x,security,label='Тревога безопасности',color='#245f73')
axes[0,0].bar(x,extra,bottom=security,label='Только недоступность',color='#d5a14d')
for i,(a,b) in enumerate(zip(security,extra)): axes[0,0].text(i,a+b+4,f'{a}+{b}',ha='center')
axes[0,0].set(xticks=x,xticklabels=methods,ylim=(0,380),ylabel='Эпизодов из 324',title='Тревоги и уведомления')
axes[0,0].legend(fontsize=9)
regimes=['reference','higher_load','cyclic','bursty_recovery']
for shift,m,color in [(-.18,'PREVIOUS','#89969c'),(.18,'CARRY','#245f73')]:
    axes[0,1].bar(x+shift,[100*overall['regimes'][r][m]['security_fpr'] for r in regimes],width=.36,label=m,color=color)
axes[0,1].set(xticks=x,xticklabels=['Исходный\nфон','Повышенная\nнагрузка','Циклическая','Частые\nвсплески'],ylabel='Ложные тревоги, % нормальных окон',title='Устойчивость к смене фона')
axes[0,1].legend()
for shift,m,color in [(-.18,'PREVIOUS','#89969c'),(.18,'CARRY','#245f73')]:
    bars=axes[1,0].bar(x+shift,[overall['episodes'][m]['growth'][v]['security'] for v in ['fast','slow','intermittent','inside_permit']],width=.36,label=m,color=color)
    axes[1,0].bar_label(bars,padding=3)
axes[1,0].set(xticks=x,xticklabels=['Быстрый','Медленный','С паузами','Внутри\nразрешения'],ylim=(0,11),ylabel='Эпизодов из 9',title='Варианты роста нагрузки')
axes[1,0].legend()
sizes=['8','16','32']; x2=np.arange(3)
unknown=[]; unavailable=[]
for a in sizes:
    v=d['by_actors'][a]['metrics']['slot-4']['CARRY']; unknown.append(100*v['unknown_normal']/v['normal_windows']); unavailable.append(100*v['availability_normal_windows']/v['normal_windows'])
axes[1,1].bar(x2,unknown,label='Unknown',color='#89969c')
axes[1,1].bar(x2,unavailable,bottom=unknown,label='Недоступность',color='#d5a14d')
axes[1,1].set(xticks=x2,xticklabels=sizes,xlabel='Число NF-позиций',ylabel='% нормальных окон slot-4',title='Отсутствие определённого контроля')
axes[1,1].legend()
for ax in axes.flat: ax.set_axisbelow(True); ax.grid(axis='y',alpha=.18)
fig.suptitle('Часовые проверки неизменных моделей на 8, 16 и 32 NF',fontsize=16)
fig.supxlabel('По три стенда на масштаб · пороги взяты из предыдущей серии без переобучения\nПрисутствие сигнала в эпизоде не доказывает причинную связь с атакой.\nНедоступность контроля не является доказательством атаки.',fontsize=10)
for ext in ['png','svg','pdf']: fig.savefig(root/f'comparison.{ext}',dpi=200,bbox_inches='tight')
