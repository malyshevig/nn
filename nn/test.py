


import torch
print (torch.__version__)

a = torch.randn(1, 240, 120)

a1 = torch.topk(input=a, k=15)
a2 = torch.topk(input=a, k=20)

a1mps = torch.topk(input=a.to("mps"), k=15) # working fine
a2mps = torch.topk(input=a.to("mps"), k=20) # not working at all

print(torch.allclose(a1mps[0].to("cpu"), a1[0])) # True
print(torch.allclose(a2mps[0].to("cpu"), a2[0])) # False

print(a2mps[0:0]) # last 5 values are always 0 not proper top sorted values values
print(a2[0:0]) # last values are fine



