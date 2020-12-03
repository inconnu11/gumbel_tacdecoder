x = torch.arange(5)   
print(x)
mask = torch.eq(x,3)   # 等于
print(mask)
print(x[mask])