import torch

from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument("--output_disp", action='store_true', help="save disparity img",default = 'output/disp/')
#parser.add_argument("--output_depth", action='store_true', help="save depth img",default = 'output/depth/')
parser.add_argument("--pretrained",  type=str, help="pretrained DispNet path",default = '/home/roit/models/SfMLearner/dispnet_model_best.pth')
#parser.add_argument("--pretrained",  type=str, help="pretrained DispNet path",default = 'checkpoints/dump/dispnet_model_best.pth')

parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='/home/roit/datasets/KITTI/raw_data/test/sq1', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg','bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    '''
    #还挑着生成哪个, 这里都输出了
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return
    '''
    disp_net = DispNetS().to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)#str2Path
    output_disp_dir = Path(args.output_dir+'/disp')
    output_depth_dir = Path(args.output_dir+'/depth')

    output_disp_dir.makedirs_p()#如果没有就创建,甚至可以创建子文件夹
    output_depth_dir.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files):#测试图片

        img = imread(file).astype(np.float32)

        h,w,_ = img.shape#h :375 w:1242 _: 3
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)

        #网络输入
        output = disp_net(tensor_img)#1,1,h,w
        output = output[0]
        file_path, file_ext = file.relpath(args.dataset_dir).splitext()
        file_name = '-'.join(file_path.splitall())



        #save to disk

        disp = (255*tensor2array(output, max_value=None, colormap='bone')).astype(np.uint8)#4x375x1242
        imsave(output_disp_dir/'{}_disp{}'.format(file_name, file_ext), np.transpose(disp, (1,2,0)))#多通道图像转至（1,2,0）,375x1242x4

        depth = 1/output
        depth = (255*tensor2array(depth, max_value=10, colormap='rainbow')).astype(np.uint8)
        imsave(output_depth_dir/'{}_depth{}'.format(file_name, file_ext), np.transpose(depth, (1,2,0))[:,:,1])


if __name__ == '__main__':
    main()
