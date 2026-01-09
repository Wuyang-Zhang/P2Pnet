import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')


from tools_usr.Count_Lines import count_lines
 
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='output/pre_result',
                        help='path where to save')
    parser.add_argument('--weight_path', default=r'ckpt\best_mae.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--test_data', default=r'crowd_datasets\SHHA\val_real_test')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    return parser

def main(args, debug=False):
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    model = build_model(args) # get the P2PNet
    model.to(device) # move to GPU
    if args.weight_path is not None: # load trained model
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model.eval() # convert to eval mode    

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    with open("./output/pre_gd_cnt.txt", 'w') as file:
        for i in os.listdir(args.test_data):
            if i.endswith(".jpg"):
                jpg_name = os.path.basename(i).split(".")[0]
                # set your image path here
                img_path = os.path.join(args.test_data,i) 
                print(img_path) 
                # set your gdtxt path here
                img_txt_path = img_path.replace(".jpg",".txt")
                print(img_txt_path) 
                # load the images
                img_raw = Image.open(img_path).convert('RGB')
                # round the size
                width, height = img_raw.size
                print("origin->width, height :",width, height)
                if width > 1080 and height > 1080 :
                    new_width =  1080 // 128 * 128
                    new_height = (height//(width // 1080))// 128 * 128
                    img_raw = img_raw.resize((new_width, new_height))
                else:
                    new_width = width // 128 * 128
                    new_height = height // 128 * 128
                    # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
                    img_raw = img_raw.resize((new_width, new_height))
                print("precessing->width, height :",img_raw.size[0], img_raw.size[1])
                # pre-proccessing
                img = transform(img_raw)
                samples = torch.Tensor(img).unsqueeze(0)
                samples = samples.to(device)
                # run inference
                outputs = model(samples)
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
                outputs_points = outputs['pred_points'][0]
                threshold = 0.5
                # filter the predictions
                points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                predict_cnt = int((outputs_scores > threshold).sum())
                print("predict_cnt",predict_cnt)
                gd_cnt = count_lines(img_txt_path)
                print("gd_cnt",gd_cnt)
                file.write(f"{jpg_name} {predict_cnt} {gd_cnt}\n")
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
                outputs_points = outputs['pred_points'][0]
                size = 2 # draw the predictions
                img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)  # is  resized images
                for p in points:
                    img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
                # save the visualized image
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                cv2.imwrite(os.path.join(args.output_dir, f'{jpg_name}_pred_{predict_cnt}.jpg'), img_to_draw)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)      
        
        
        