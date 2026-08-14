import numpy as np,json,os
C='/root/5.6+chanshui1/outputs/t2_cache_station27'
X=np.load(C+'/X.npy',mmap_mode='r');Y=np.load(C+'/y.npy');N=np.load(C+'/net.npy').astype(str);S=np.load(C+'/sta.npy').astype(str)
assert len(X)==len(Y)==len(N)==len(S) and len(X)>50000
sid=np.char.add(np.char.add(N,'.'),S)
json.dump(dict(n=int(len(X)),protocol='STEAD P-5s..P+5s',networks=int(len(set(N.tolist()))),stations=int(len(set(sid.tolist()))),verified_shapes=dict(X=list(X.shape),y=list(Y.shape),net=list(N.shape),sta=list(S.shape))),open(C+'/meta.json','w'),indent=2)
print(open(C+'/meta.json').read())
