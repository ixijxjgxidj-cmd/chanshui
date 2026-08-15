import os
os.environ['PRE_EPOCHS']='1'; os.environ['PRE_CAP']='1500'
src=open('/root/5.6+chanshui1/train39c_ablation.py').read()
src=src.replace("SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),\n          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]", "SETUPS = [('CHILE','ALASKA','NZ')]")
src=src.replace("GRID = [dict(tag='p8_m32_b10', patch=8, mask_ratio=.32, blk=10, ctx_keep=.60),\n        dict(tag='p8_m48_b10', patch=8, mask_ratio=.48, blk=10, ctx_keep=.45),\n        dict(tag='p8_m64_b10', patch=8, mask_ratio=.64, blk=10, ctx_keep=.30),\n        dict(tag='p8_m48_b25', patch=8, mask_ratio=.48, blk=25, ctx_keep=.45),\n        dict(tag='p16_m48_b6', patch=16, mask_ratio=.48, blk=6, ctx_keep=.45),\n        dict(tag='p32_m48_b4', patch=32, mask_ratio=.48, blk=4, ctx_keep=.45)]", "GRID = [dict(tag='p8_m48_b10', patch=8, mask_ratio=.48, blk=10, ctx_keep=.45)]")
src=src.replace('def finetune(enc, tr_rows, seed, epochs=8', 'def finetune(enc, tr_rows, seed, epochs=1')
src=src.replace("OUT = f'{ROOT}/outputs/t2_round39c_ablation'", "OUT = f'{ROOT}/outputs/t2_round39c_smoke'")
exec(compile(src,'smoke39c','exec'), {'__name__':'__main__'})