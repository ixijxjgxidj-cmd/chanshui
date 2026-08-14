"""公开 STEAD P±5s 缓存，保留逐样本 network.station 血缘。
只允许公开 STEAD；不读取比赛目录。
"""
import os,time,json
import numpy as np,pandas as pd
import seisbench.data as sbd
from t2_io import to_enz_from_stead
ROOT='/root/5.6+chanshui1';OUT=f'{ROOT}/outputs/t2_cache_station27';os.makedirs(OUT,exist_ok=True)
d=sbd.STEAD();md=d.metadata;m=pd.to_numeric(md['source_magnitude'],errors='coerce');tp=pd.to_numeric(md['trace_p_arrival_sample'],errors='coerce');base=tp.notna()&m.notna();hi=md.index[base&(m>=4)].to_numpy();lo=md.index[base&(m>=3.2)&(m<4)].to_numpy().copy();rs=np.random.RandomState(20260815);rs.shuffle(lo);lo=lo[:45000];idx=np.concatenate([hi,lo]);rs.shuffle(idx)
WIN=1000;X=np.zeros((len(idx),3,WIN),np.float32);Y=np.zeros(len(idx),np.float32);NET=np.empty(len(idx),object);STA=np.empty(len(idx),object);SRC=np.empty(len(idx),object);k=0;sk=0;t0=time.time()
for j,i0 in enumerate(idx):
 i=int(i0)
 try:wv=d.get_waveforms(i)
 except Exception:sk+=1;continue
 row=md.iloc[i];p=int(tp.iloc[i]);w=to_enz_from_stead(wv,row.get('trace_component_order','ZNE'))
 if p<500 or w.shape[1]-p<500:sk+=1;continue
 seg=w[:,p-500:p+500];seg=(seg-seg.mean(1,keepdims=True)).astype(np.float32)
 X[k]=seg;Y[k]=float(m.iloc[i]);NET[k]=str(row.get('station_network_code','NA'));STA[k]=str(row.get('station_code','NA'));SRC[k]=str(row.get('source_id','NA'));k+=1
 if k%5000==0:print(k,'/',len(idx),time.time()-t0,flush=True)
X=X[:k];Y=Y[:k];NET=NET[:k].astype(str);STA=STA[:k].astype(str);SRC=SRC[:k].astype(str);np.save(f'{OUT}/X.npy',X);np.save(f'{OUT}/y.npy',Y);np.save(f'{OUT}/net.npy',NET);np.save(f'{OUT}/sta.npy',STA);np.save(f'{OUT}/src.npy',SRC)
json.dump(dict(n=int(k),skipped=int(sk),protocol='STEAD P-5s..P+5s',networks=int(len(set(NET))),stations=int(len(set(NET+'.'+STA)))),open(f'{OUT}/meta.json','w'),indent=2)
print('DONE',X.shape,'networks',len(set(NET)),'stations',len(set(NET+'.'+STA)),flush=True)
