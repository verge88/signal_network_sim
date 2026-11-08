"""Scientific figure from measured pooled output; no generated data."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
data = json.loads((root/'summary.json').read_text())
names = ['H','O','K']
cases = ['S1','S2','S3','S4','G','C','F','R','D']
labels = ['Профиль NRF','Запрещённое действие','Маршрут','Всплеск с ошибками',
          'Постепенный рост','Конфигурация','Конфигурация + подделка','Повтор отчёта','Подавление отчёта']
matrix = np.array([[data['episode_summary'][m]['scenarios'][s]['detected'] /
                    data['episode_summary'][m]['scenarios'][s]['n'] for m in names] for s in cases])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig, (left,right) = plt.subplots(1,2,figsize=(12,6.6),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
left.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
left.set_yticks(range(len(cases)),labels)
left.set_xticks(range(3),['Гибрид H','H + наблюдение O','H + защита K'],rotation=15,ha='right')
for i,s in enumerate(cases):
    for j,m in enumerate(names):
        item = data['episode_summary'][m]['scenarios'][s]
        left.text(j,i,f"{item['detected']}/{item['n']}",ha='center',va='center',color='white' if matrix[i,j]>.6 else '#172e39')
left.set_title('Обнаруженные эпизоды; пропуски сохранены')
unknown = [100*(1-data['metrics']['slot-4'][m]['normal_coverage']) for m in ['O','K']]
bars=right.bar(['O','K'],unknown,color=['#69a495','#256c61'])
right.bar_label(bars,labels=[f'{v:.2f}%' for v in unknown],padding=5)
right.set_ylabel('% нормальных окон slot-4')
right.set_title('Нет достоверного нового наблюдения\nТолько NF с новым контрольным каналом')
right.set_ylim(0,max(15,max(unknown)*1.3))
right.spines[['top','right']].set_visible(False)
fig.suptitle('Совместная проверка гибрида и независимого КИЕ',fontsize=16)
fig.supxlabel('3 стенда × 20 минут теста · H / O / K на одинаковых событиях\nUnknown не считается обнаружением атаки или подтверждением нормального состояния.',fontsize=10)
for extension in ['png','svg','pdf']:
    fig.savefig(root/f'integrated_comparison.{extension}',dpi=200,bbox_inches='tight')
