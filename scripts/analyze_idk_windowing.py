import pandas as pd

df = pd.read_csv('results/merged/idk_windowing_compare/idk_windowing_summary.csv')

print('=== 各数据集滑窗IDK与直接IDK对比 ===')
for dataset in df['dataset'].unique():
    print(f'\n【{dataset}】')
    sub = df[df['dataset'] == dataset]
    direct = sub[sub['mode'] == 'direct']
    sliding = sub[sub['mode'] == 'sliding']
    
    if not direct.empty:
        d_ari = float(direct['ari_mean'].iloc[0])
        d_nmi = float(direct['nmi_mean'].iloc[0])
        d_rt = float(direct['runtime_mean'].iloc[0])
        print(f'  direct: ARI={d_ari:.4f}, NMI={d_nmi:.4f}, runtime={d_rt:.2f}s')
    
    best_ari = -1
    best_nmi = -1
    best_win = ''
    for _, row in sliding.iterrows():
        win = row['window_label']
        s_ari = float(row['ari_mean'])
        s_nmi = float(row['nmi_mean'])
        s_rt = float(row['runtime_mean'])
        print(f'  {win}: ARI={s_ari:.4f}, NMI={s_nmi:.4f}, runtime={s_rt:.2f}s')
        if s_ari > best_ari:
            best_ari = s_ari
            best_nmi = s_nmi
            best_win = win
    
    if not direct.empty and best_win:
        ari_gain = best_ari - d_ari
        nmi_gain = best_nmi - d_nmi
        print(f'  -> 最佳滑窗{best_win}相比direct: ARI提升={ari_gain:.4f}, NMI提升={nmi_gain:.4f}')

print('\n=== 汇总统计 ===')
print(df.groupby(['dataset', 'mode']).agg({
    'ari_mean': 'mean',
    'nmi_mean': 'mean',
    'runtime_mean': 'mean'
}).reset_index().to_string())
