import torch
import torchvision

from data_loader import MyDataset
from model import RegressionModel
from functions import increase_saturation

from torch.utils.data import DataLoader
from IQA_pytorch import SSIM, MS_SSIM, CW_SSIM, GMSD, LPIPSvgg, DISTS, NLPD, FSIM, VSI, VIFs, VIF, MAD
from kornia.metrics import psnr

import numpy as np
import os

# img_size = 512
# target_shape = (img_size, img_size)
target_shape = None
batch_size = 1

# test dataset
test_input = '/data/agc/zero_dce/test-input'
# '/data/agc/LCDPNet/src/ckpt/test_result/lcdpnet_ckpt_trained_on_ours.ckpt@lcdp_data.test.512'
# '/data/agc/zero_dce/test-input'
# '/data/agc/EnlightenGAN/test_200/images'
# '/data/agc/Illumination-Adaptive-Transformer/dataset/exposure/validation/Result'

# 텐서 데이터 불러오기
test_data = MyDataset(test_input, target_shape=target_shape, mode='val')
print("테스트 데이터셋 개수", test_data.len)

# 미니 배치 형태로 데이터 갖추기
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)
print("배치 개수",len(test_loader))

# CPU/GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

n = len(test_loader) # 배치 개수

ssim = SSIM()
ms_ssim = MS_SSIM()
cw_ssim = CW_SSIM()

# fsim = FSIM()
# vsi = VSI()
gmsd = GMSD().to(device)

nlpd_4 = NLPD(k=4).to(device)
nlpd_6 = NLPD(k=6).to(device)
# mad = MAD()

vifs = VIFs()
# vif = VIF()

lpips = LPIPSvgg().to(device)
dists = DISTS().to(device)

psnr_list = []

ssim_list = []
ms_ssim_list = []
cw_ssim_list = []

fsim_list = []
vsi_list = []
gmsd_list = []

nlpd_4_list = []
nlpd_6_list = []
mad_list = []

vifs_list = []
vif_list = []

lpips_list = []
dists_list = []

for idx, imgs in enumerate(test_loader):
    outputs, gt = imgs[0].to(device), imgs[1].to(device)

    # torchvision.utils.save_image(outputs, result_path + str(name))
    # torchvision.utils.save_image(outputs, result_path + str(idx) + '.png')
    # torchvision.utils.save_image(high_img, result_gt_path + str(i) + '.png')
    # psnr_value = psnr(enhanced_img, high_img).item()

    val_psnr = psnr(outputs, gt, 1.0)

    val_ssim = ssim(outputs, gt, as_loss=False).item()
    val_ms_ssim = ms_ssim(outputs, gt, as_loss=False).item()
    val_cw_ssim = cw_ssim(outputs, gt, as_loss=False).item()

    # val_fsim = fsim(outputs, gt, as_loss=False).item()
    # val_vsi = vsi(outputs, gt, as_loss=False).item()
    val_gmsd = gmsd(outputs, gt, as_loss=False).item()

    val_nlpd_4 = nlpd_4(outputs, gt, as_loss=False).item()
    val_nlpd_6 = nlpd_6(outputs, gt, as_loss=False).item()
    # val_mad = mad(outputs, gt, as_loss=False).item()

    val_vifs = vifs(outputs, gt, as_loss=False).item()
    # val_vif = vif(outputs, gt, as_loss=False).item()

    val_lpips = lpips(outputs, gt, as_loss=False).item()
    val_dists = dists(outputs, gt, as_loss=False).item()
    
    psnr_list.append(val_psnr.cpu())
    
    ssim_list.append(val_ssim)
    ms_ssim_list.append(val_ms_ssim)
    cw_ssim_list.append(val_cw_ssim)
    
    # fsim_list.append(val_fsim)
    # vsi_list.append(val_vsi)
    gmsd_list.append(val_gmsd)

    nlpd_4_list.append(val_nlpd_4)
    nlpd_6_list.append(val_nlpd_6)
    # mad_list.append(val_mad)

    vifs_list.append(val_vifs)
    # vif_list.append(val_vif)

    lpips_list.append(val_lpips)
    dists_list.append(val_dists)

PSNR_mean = np.mean(psnr_list)

SSIM_mean = np.mean(ssim_list)
MS_SSIM_mean = np.mean(ms_ssim_list)
CW_SSIM_mean = np.mean(cw_ssim_list)

# FSIM_mean = np.mean(fsim_list)
# VSI_mean = np.mean(vsi_list)
GMSD_mean = np.mean(gmsd_list)

NLPD_4_mean = np.mean(nlpd_4_list)
NLPD_6_mean = np.mean(nlpd_6_list)
# MAD_mean = np.mean(mad_list)

VIFS_mean = np.mean(vifs_list)
# VIF_mean = np.mean(vif_list)

LPIPS_mean = np.mean(lpips_list)
DISTS_mean = np.mean(dists_list)

# total_score = SSIM_mean + PSNR_mean + (1-LPIPS_mean)

print('The PSNR Value is:', PSNR_mean)

print('The SSIM Value is:', SSIM_mean)
print('The MS_SSIM Value is:', MS_SSIM_mean)
print('The CW_SSIM Value is:', CW_SSIM_mean)

# print('The FSIM Value is:', FSIM_mean)
# print('The VSI Value is:', VSI_mean)
print('The GMSD Value is:', GMSD_mean)

print('The NLPD_4 Value is:', NLPD_4_mean)
print('The NLPD_6 Value is:', NLPD_6_mean)
# print('The MAD Value is:', MAD_mean)

print('The VIFs Value is:', VIFS_mean)
# print('The VIF Value is:', VIF_mean)

print('The LPIPS Value is:', LPIPS_mean)
print('The DISTS Value is:', DISTS_mean)


# if total_score > total_score_threshold or SSIM_mean >= 0.84 or PSNR_mean >= 21.8:
#     f = open("/data/agc/agc_new/results.txt", 'a')
#     f.write("agc_new_"+weights+'\n')
#     f.write(f'The total score is:{total_score}\n')
#     f.write(f'The SSIM Value is:{SSIM_mean}\n')
#     f.write(f'The PSNR Value is:{PSNR_mean}\n')
#     f.write(f'The LPIPS Value is:{LPIPS_mean}\n')
#     f.close()
#     if total_score > total_score_threshold:
#         total_score_threshold = total_score
