import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
data=json.loads((root/'confirmation_summary.json').read_text(encoding='utf-8'))
rows={r['method']:r for r in data['metrics']}
names=['rules','IF_CONTEXT','hybrid','robust_tail','hybrid_tail']
labels=['Правила\nпрофиля','IF\nс контекстом','Исходный\nгибрид','Статистика\nнормы','Новый\nгибрид']
colors=['#96a0aa','#698dab','#698dab','#37a58a','#176b56']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
x=range(len(names))
for ax,key,title in [(axes[0],'recall','Доля обнаруженных атакующих NF-окон'),(axes[1],'fpr','Доля ложных тревог на нормальных NF-окнах')]:
    values=[100*rows[n][key] for n in names]
    bars=ax.bar(x,values,color=colors,width=.65)
    ax.set_xticks(list(x),labels,fontsize=10)
    ax.set_title(title,fontsize=12,pad=14)
    ax.set_ylabel('%')
    ax.set_axisbelow(True); ax.grid(axis='y',alpha=.22)
    ax.bar_label(bars,labels=[f'{v:.3f}%' if key=='fpr' else f'{v:.2f}%' for v in values],padding=5,fontsize=10)
axes[0].set_ylim(0,114)
axes[1].set_ylim(0,.122)
axes[1].axhline(.1,color='#a75e36',linestyle='--',linewidth=1.2)
axes[1].text(4.35,.102,'Ориентир 0,1%',ha='right',fontsize=9,color='#91502c')
fig.suptitle('Подтверждающая серия на Open5GS 2.7.6',fontsize=16)
fig.supxlabel('9 запусков · 40 320 NF-окон · 36 эпизодов\nНаблюдаемые частоты лабораторного опыта; гарантия для операторской сети не устанавливается.',fontsize=10)
out=root
for ext in ('png','svg','pdf'):
    fig.savefig(out/f'ablation_v2.{ext}',dpi=200,bbox_inches='tight')
print(out/'ablation_v2.png')
