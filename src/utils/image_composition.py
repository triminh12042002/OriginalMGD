import torch
import torchvision.transforms.functional as F

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
transform = T.ToPILImage()

def compose_img(gt_img, fake_img, im_parse):

    seg_head = torch.logical_or(im_parse == 1, im_parse == 2)
    seg_head = torch.logical_or(seg_head, im_parse == 4)
    seg_head = torch.logical_or(seg_head, im_parse == 13)

    # img = transform(fake_img)
    # img.save("fake_img.jpg")

    
    

    true_head = gt_img * seg_head
    true_parts = true_head

    # img = transform(true_head)
    # img.save("true_head.jpg")
    
    generated_body = (F.pil_to_tensor(fake_img).cuda() / 255) * (~(seg_head))

    return true_parts + generated_body

def compose_img_dresscode(gt_img, fake_img, im_head):

    seg_head = im_head
    true_head = gt_img * seg_head
    generated_body = fake_img * ~(seg_head)

    return true_head + generated_body 