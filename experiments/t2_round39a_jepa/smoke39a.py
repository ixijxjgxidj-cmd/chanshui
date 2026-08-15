"""轮39A 冒烟测试：单组 A->B->C、1 epoch 预训练/微调，仅验证代码路径与合规断言。"""
import os
os.environ.setdefault('PRE_EPOCHS', '1')
os.environ.setdefault('PRE_CAP', '3000')
os.environ.setdefault('ARMS', 'cnn,scratch,jepa_a,jepa_mr')
src = open('/root/5.6+chanshui1/train39a_jepa.py').read()
src = src.replace("SETUPS = [('ALASKA', 'CALIF', 'GREECE'), ('CALIF', 'GREECE', 'ALASKA'), ('GREECE', 'CHILE', 'CALIF'),\n          ('CHILE', 'ALASKA', 'NZ'), ('OTHER', 'NZ', 'CHILE')]",
                  "SETUPS = [('CHILE', 'ALASKA', 'NZ')]")
src = src.replace('def finetune(enc, tr_rows, seed, epochs=8', 'def finetune(enc, tr_rows, seed, epochs=1')
src = src.replace('def train_cnn(tr_rows, seed, epochs=8', 'def train_cnn(tr_rows, seed, epochs=1')
src = src.replace("OUT = f'{ROOT}/outputs/t2_round39a_jepa'", "OUT = f'{ROOT}/outputs/t2_round39a_smoke'")
exec(compile(src, 'smoke39a', 'exec'), {'__name__': '__main__'})