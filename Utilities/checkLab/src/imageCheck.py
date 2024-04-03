import argparse
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import functional as F

from django.conf import settings
import os

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from Utilities.checkLab.src.config import update_config
from Utilities.checkLab.src.config import _C as config
from Utilities.checkLab.src.data_core import myDataset

parser = argparse.ArgumentParser(description='Morphed Image Detection')
parser.add_argument('-gpu', '--gpu', type=int, default=-1, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default='../output', help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)


imageFile = ''

def setPath(imagepath):
    global imageFile
    imageFile = imagepath


def check():
    print(imageFile)
    return test()

def test():

    device = 'cpu'
    np.set_printoptions(formatter={'float': '{: 7.3f}'.format})


    list_img = [imageFile]

    print(list_img)

    test_dataset = myDataset(list_img=list_img)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1)  # 1 to allow arbitrary imageFile sizes

    model_state_file = r'Utilities/checkLab\weights\trufor.pth.tar'

    print('=> loading model from {}'.format(model_state_file))

    # Load the model with appropriate mapping based on CUDA availability
    if torch.cuda.is_available():
        checkpoint = torch.load(model_state_file)
    else:
        checkpoint = torch.load(model_state_file, map_location=torch.device('cpu'))

    if config.MODEL.NAME == 'detconfcmx':
        from Utilities.checkLab.src.models.cmx.builder_np_conf import myEncoderDecoder as confcmx
        model = confcmx(cfg=config)
    else:
        raise NotImplementedError('Model not implemented')

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    with torch.no_grad():
        for index, (rgb, path) in enumerate(tqdm(testloader)):
            
            filename_out = imageFile + '.npz'

            if not (os.path.isfile(filename_out)):
                try:
                    rgb = rgb.to(device)
                    model.eval()

                    det = None
                    conf = None

                    pred, conf, det, npp = model(rgb)

                    if conf is not None:
                        conf = torch.squeeze(conf, 0)
                        conf = torch.sigmoid(conf)[0]
                        conf = conf.cpu().numpy()

                    if npp is not None:
                        npp = torch.squeeze(npp, 0)[0]
                        npp = npp.cpu().numpy()

                    if det is not None:
                        det_sig = torch.sigmoid(det).item()

                    pred = torch.squeeze(pred, 0)
                    pred = F.softmax(pred, dim=0)[1]
                    pred = pred.cpu().numpy()

                    out_dict = dict()
                    out_dict['map'] = pred
                    out_dict['imgsize'] = tuple(rgb.shape[2:])
                    if det is not None:
                        out_dict['score'] = det_sig
                    if conf is not None:
                        out_dict['conf'] = conf

                    out_dict['np++'] = npp
                    print(imageFile.split('/')[2])
                    plt.imshow(out_dict['map'], cmap='RdBu_r', clim=[0,1])  # Choose the colormap according to your data
                    plt.axis('off')
                    plt.imsave(rf'media/result-image/{imageFile.split('/')[2]}', out_dict['map'], cmap='RdBu_r')
                            
                except:
                    pass
                            
        return {"prediction": 1 if out_dict['score'] >= 0.5 else 0, "confidence": "{:.2f}%".format(out_dict['score'] * 100)} 