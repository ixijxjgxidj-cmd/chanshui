"""形状回归测试：三种 patch 的 tokenizer 输出必须与预设 token 数一致，掩码索引不得越界。"""
import sys, numpy as np, torch
sys.path.insert(0, '/root/5.6+chanshui1')
src = open('/root/5.6+chanshui1/train39c_ablation.py').read()
cut = src.index('SETUPS = [')
ns = {'__name__': 'shapecheck'}
exec(compile(src[:cut], 'shapecheck', 'exec'), ns)
N_TOK = {8: 125, 16: 63, 32: 32}
fail = []
for patch, nt in N_TOK.items():
    tok = ns['Tokenizer'](patch)
    x = torch.randn(4, 3, 1000)
    t = tok(x)
    print('patch', patch, 'tokens', t.shape[1], 'expect >=', nt)
    if t.shape[1] < nt:
        fail.append((patch, t.shape[1], nt))
    enc = ns['Encoder'](patch, nt)
    out = enc(x)
    assert out.shape == (4, nt, 128), (patch, out.shape)
    rng = np.random.RandomState(0)
    ctx, tgt = ns['sample_masks'](4, rng, nt, .48, min(10, nt // 4), .45)
    assert int(ctx.max()) < nt and int(tgt.max()) < nt, (patch, int(ctx.max()), int(tgt.max()))
    keep = enc(x, ctx)
    pred = ns['Predictor'](nt)(keep, ctx, tgt)
    assert pred.shape == (4, tgt.shape[1], 128), (patch, pred.shape)
    print('  OK enc', tuple(out.shape), 'ctx', tuple(ctx.shape), 'tgt', tuple(tgt.shape), 'pred', tuple(pred.shape))
if fail:
    raise SystemExit(f'token 数不足: {fail}')
print('SHAPE_ALL_OK')