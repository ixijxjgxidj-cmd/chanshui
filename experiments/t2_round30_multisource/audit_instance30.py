import pandas as pd,json,os,numpy as np
p='/root/5.6+chanshui1/public_round30/instance_metadata.csv.bz2'
use=['source_magnitude','source_magnitude_type','trace_dt_s','trace_npts','trace_P_arrival_sample','trace_S_arrival_sample','station_network_code','station_code','source_id','source_type','trace_deconvolved_units','path_ep_distance_km']
df=pd.read_csv(p,usecols=use,compression='bz2')
m=pd.to_numeric(df.source_magnitude,errors='coerce');dt=pd.to_numeric(df.trace_dt_s,errors='coerce');pp=pd.to_numeric(df.trace_P_arrival_sample,errors='coerce');nn=pd.to_numeric(df.trace_npts,errors='coerce')
ok=m.notna()&dt.notna()&pp.notna()&nn.notna()&(pp*dt>=5)&((nn-pp)*dt>=5)
rep=dict(n_total=int(len(df)),n_mag=int(m.notna().sum()),n_p10_eligible=int(ok.sum()),magnitude=dict(q={str(q):float(m.quantile(q)) for q in [.01,.1,.25,.5,.75,.9,.99]},min=float(m.min()),max=float(m.max())),dt_counts={str(k):int(v) for k,v in dt.value_counts().head(10).items()},mag_types={str(k):int(v) for k,v in df.source_magnitude_type.value_counts().head(12).items()},networks=int(df.station_network_code.nunique()),stations=int(df.station_code.nunique()),source_types={str(k):int(v) for k,v in df.source_type.value_counts().items()},units={str(k):int(v) for k,v in df.trace_deconvolved_units.value_counts().items()})
os.makedirs('/root/5.6+chanshui1/outputs/t2_round30',exist_ok=True);json.dump(rep,open('/root/5.6+chanshui1/outputs/t2_round30/instance_metadata_audit.json','w'),indent=2);print(json.dumps(rep,indent=2))
