import torch
import torchvision

from data_loader import EvaluationDataset
from model import RegressionModel
from functions import increase_saturation

from torch.utils.data import DataLoader
from IQA_pytorch import SSIM, LPIPSvgg, DISTS
from kornia.metrics import psnr

import numpy as np
import os

img_size = 512

path_1 = '/data/agc/agc_new/outputs/agc_new_61'
path_2 = '/data/agc/agc_new/outputs/lcdp'
path_3 = '/data/agc/Illumination-Adaptive-Transformer/dataset/exposure/validation/Result'
path_4 = '/data/agc/agc_new/outputs/EnlightenGAN'
path_5 = '/data/agc/agc_new/outputs/zero_dce'
gt_path = '/data/agc/agc_new/outputs/gt'

# 텐서 데이터 불러오기
test_data = EvaluationDataset(path_1, path_2, path_3, path_4, path_5, gt_path)
print("테스트 데이터셋 개수", test_data.len)

# 미니 배치 형태로 데이터 갖추기
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16)
print("배치 개수",len(test_loader))

# CPU/GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

n = len(test_loader) # 배치 개수

ssim = SSIM()
lpips = LPIPSvgg().to(device)
# dists = DISTS().to(device)

ssim_list = []
psnr_list = []
lpips_list = []
dists_list = []


for idx, imgs in enumerate(test_loader):
    filename, ours, lcdp, iat, engan, zero, gt = imgs[0], imgs[1].to(device), imgs[2].to(device), imgs[3].to(device), imgs[4].to(device), imgs[5].to(device), imgs[6].to(device)
    # inputs, gt, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
    print(filename)

    # ssim_value = ssim(outputs, gt, as_loss=False).item()
    # # psnr_value = psnr(enhanced_img, high_img).item()
    # val_psnr = psnr(outputs, gt, 1.0)
    # val_lpips = lpips(outputs, gt, as_loss=False).item()
    # val_dists = dists(outputs, gt, as_loss=False).item()
    
    our_lpips = lpips(ours, gt, as_loss=False).item()
    our_psnr = psnr(ours, gt, 1.0)
    our_ssim = ssim(ours, gt, as_loss=False).item()

    lcdp_lpips = lpips(lcdp, gt, as_loss=False).item()
    lcdp_psnr = psnr(lcdp, gt, 1.0)
    lcdp_ssim = ssim(lcdp, gt, as_loss=False).item()

    iat_lpips = lpips(iat, gt, as_loss=False).item()
    iat_psnr = psnr(iat, gt, 1.0)
    iat_ssim = ssim(iat, gt, as_loss=False).item()

    engan_lpips = lpips(engan, gt, as_loss=False).item()
    engan_psnr = psnr(engan, gt, 1.0)
    engan_ssim = ssim(engan, gt, as_loss=False).item()

    zero_lpips = lpips(zero, gt, as_loss=False).item()
    zero_psnr = psnr(zero, gt, 1.0)
    zero_ssim = ssim(zero, gt, as_loss=False).item()

    # ours의 lpips가 다른 lpips 중 가장 작을 때 lpips, psnr, ssim 비교
    if our_lpips < lcdp_lpips and our_lpips < iat_lpips and our_lpips < engan_lpips and our_lpips < zero_lpips and our_psnr < lcdp_psnr and our_ssim < lcdp_ssim:
        f = open("/data/agc/agc_new/lpips_results.txt", 'a')
        f.write("file number:"+str(filename)+"   gap:"+str(lcdp_lpips-our_lpips)+'\n')
        f.write(f'our_lpips:{our_lpips}, {our_psnr}, {our_ssim}\n')
        f.write(f'lcdp_lpips:{lcdp_lpips}, {lcdp_psnr}, {lcdp_ssim}\n')
        f.write(f'iat_lpips:{iat_lpips}, {iat_psnr}, {iat_ssim}\n')
        f.write(f'engan_lpips:{engan_lpips}, {engan_psnr}, {engan_ssim}\n')
        f.write(f'zero_lpips:{zero_lpips}, {zero_psnr}, {zero_ssim}\n')
        f.close()



    # print('The SSIM Value is:', SSIM_mean)
    # print('The PSNR Value is:', PSNR_mean)
    # print('The LPIPS Value is:', LPIPS_mean)
    # if total_score > total_score_threshold or SSIM_mean >= 0.84 or PSNR_mean >= 21.8:
    # f = open("/data/agc/agc_new/results.txt", 'a')
    # f.write("agc_new_"+weights+'\n')
    # f.write(f'The total score is:{total_score}\n')
    # f.write(f'The SSIM Value is:{SSIM_mean}\n')
    # f.write(f'The PSNR Value is:{PSNR_mean}\n')
    # f.write(f'The LPIPS Value is:{LPIPS_mean}\n')
    # f.write(f'The DISTS Value is:{DISTS_mean}\n')
    # f.close()
    # if total_score > total_score_threshold:
        # total_score_threshold = total_score
