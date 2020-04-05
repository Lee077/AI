import torch
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

model= torch.load('./results/coco/checkpoint_epoch_1.pth.tar')['model']


def show_and_tell(filename, beam_size=3):
    img = Image.open(filename, 'r')
    imshow(np.asarray(img))
    captions = model.generate(img, beam_size=beam_size)
    print(captions,'===========')

# show_and_tell('./COCO_val2014_000000000073.jpg')
show_and_tell('/home/lin7u/Downloads/dataset/coco2014/val2014/COCO_val2014_000000000073.jpg')
show_and_tell('/home/lin7u/Downloads/dataset/coco2014/val2014/COCO_val2014_000000290957.jpg')

#1
#['EOS EOS', 'a EOS', 'diagonal EOS'] ===========
#['EOS EOS', 'a EOS', 'elephants EOS'] ===========

#2
#['EOS EOS', 'a EOS', 'UNK EOS'] ===========
#['EOS EOS', 'a EOS', 'UNK EOS'] ===========
