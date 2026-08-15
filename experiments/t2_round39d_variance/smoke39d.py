import os
os.environ['PRE_EPOCHS']='1'; os.environ['PRE_CAP']='1200'
src=open('/root/5.6+chanshui1/train39d_variance.py').read()
src=src.replace("SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),\n          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]","SETUPS = [('CHILE','ALASKA','NZ')]")
src=src.replace('SEEDS = [11, 2029, 70707]','SEEDS = [11, 2029]')
src=src.replace("OUT = f'{ROOT}/outputs/t2_round39d_variance'","OUT = f'{ROOT}/outputs/t2_round39d_smoke'")
exec(compile(src,'smoke39d','exec'),{'__name__':'__main__'})