import torch
import numpy as np

a = []
torch.tensor([])

num_bins = 256
# x = [1,2,3,4]
x = [0.71,0.82,0.3,0.4]
# x = np.array(x)

# =================
# quantize fo numpy
# =================

# x is logf0
assert x.ndim == 1
# shape 的数字有几个，ndim 就等于几
x = x.astype(float).copy()
uv = (x <= 0)
x[uv] = 0.0
assert (x >= 0).all() and (x <= 1).all()
x = np.round(x * (num_bins - 1))
x = x + 1
x[uv] = 0.0
enc = np.zeros((len(x), num_bins + 1), dtype=np.float32)
enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
print(enc.shape)
print(enc)
print(x.astype(np.int64))
# return enc, x.astype(np.int64)

# =================
# quantize fo torch
# =================
# x is logf0
x = torch.tensor(x)
B = x.size(0)
# print(B)    = 4
x = x.view(-1).clone()
# view 相当于 reshape
uv = (x <= 0)
x[uv] = 0
assert (x >= 0).all() and (x <= 1).all()
# 条件为true时正常运行，否则触发异常
# [0,1]之间
x = torch.round(x * (num_bins - 1))
# [0,255]
x = x + 1
# 对每一个元素 +1？
# x [1,256]之间
x[uv] = 0
enc = torch.zeros((x.size(0), num_bins + 1), device=x.device)
enc[torch.arange(x.size(0)), x.long()] = 1
print(enc.view(B, -1, num_bins + 1))
print(x.view(B, -1).long())
# return enc.view(B, -1, num_bins + 1), x.view(B, -1).long()
