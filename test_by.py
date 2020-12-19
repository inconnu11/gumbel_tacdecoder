import torch

#
alignment = torch.rand(3, 3)
print(alignment)
score_mask_value = -22
mask = torch.ones([3, 1])
# print(alignment.data)
# cc = alignment.data.masked_fill_(mask, score_mask_value)
cc = alignment.data.masked_fill_(mask, score_mask_value)
print(alignment)
print(cc)
print(mask)


# a = torch.tensor([1, 0, 2, 3])
# # a.masked_fill(mask=torch.ByteTensor([1, 1, 0, 0]), value=torch.tensor(-1e9))
# a.masked_fill(mask=torch.ByteTensor([True, True, False, False]), value=torch.tensor(-1e9))