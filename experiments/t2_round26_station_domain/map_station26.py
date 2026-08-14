"""为公开缓存回接真实台站/网络标识（只读 STEAD 公开元数据）。"""
import numpy as np, pandas as pd, json, collections
import seisbench.data as sbd
ROOT='/root/5.6+chanshui1'
G=np.load(f'{ROOT}/outputs/t2_cache_p10/g.npy').astype(str)
Y=np.load(f'{ROOT}/outputs/t2_cache_p10/y.npy')
d=sbd.STEAD(); md=d.metadata
md['sid']=md['source_id'].astype(str)
# 每个 source_id 可能对应多台站；缓存只存了 source_id，故取该事件下台站的多重集用于分组一致性检查
grp=md.groupby('sid')[['station_network_code','station_code']].agg(lambda s: list(s)[:1][0])
net=grp['station_network_code'].to_dict(); sta=grp['station_code'].to_dict()
N=np.asarray([net.get(s,'NA') for s in G]); S=np.asarray([f"{net.get(s,'NA')}.{sta.get(s,'NA')}" for s in G])
np.save(f'{ROOT}/outputs/t2_cache_p10/net.npy',N); np.save(f'{ROOT}/outputs/t2_cache_p10/sta.npy',S)
m=(Y>=3.8)&(Y<=6.5)
cn=collections.Counter(N[m]); cs=collections.Counter(S[m])
big_n=[(k,v) for k,v in cn.most_common(12)]
big_s=[(k,v) for k,v in cs.most_common(12)]
print('records in range', int(m.sum()))
print('networks', len(cn), 'stations', len(cs))
print('top networks', big_n)
print('top stations', big_s)
json.dump(dict(n_in_range=int(m.sum()),n_networks=len(cn),n_stations=len(cs),top_networks=big_n,top_stations=big_s),
          open(f'{ROOT}/outputs/t2_round26_groups.json','w'),indent=2)
