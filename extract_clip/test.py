import clip as clip
import torch
from collections import OrderedDict


model, _ = clip.load("/yszhuo/model_zoo/CLIP/ViT-B-16.pt", device='cpu')
new_state_dict = OrderedDict()
for k, v in model.state_dict().items():
    if 'visual.' in k:
        if k[7:] not in ["proj", "ln_post.weight", "ln_post.bias"]:
            new_state_dict[k[7:]] = v
torch.save(new_state_dict, 'vit_b16.pth')
print(1)