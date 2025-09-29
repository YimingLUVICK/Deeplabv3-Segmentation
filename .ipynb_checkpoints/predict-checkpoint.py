import sys
import numpy as np
from PIL import Image
import torch
import torch_directml
from torchvision import transforms

from train import build_model
import utils

def main(img_path):
    dml_device = torch_directml.device()
    model = build_model().to(dml_device)
    ckpt = torch.load("best_model.pth", map_location=dml_device)
    model.load_state_dict(ckpt['model'])
    
    with Image.open(img_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im_resized, meta = utils.resize(im, label=False) 
    img_tensor = transforms.ToTensor()(im_resized)
    x = img_tensor.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        x = x.to(dml_device)
        out = model(x)["out"]
        pred = out.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

    pred = utils.unresize(pred, meta)
    pred = utils.idmask_to_rgb(pred, utils.id2color)
    result = Image.fromarray(pred)
    result.save(('result' + img_path.split('\\')[-1]))

    return result
    

if __name__ == "__main__":
    img_path = sys.argv[1]
    main(img_path)