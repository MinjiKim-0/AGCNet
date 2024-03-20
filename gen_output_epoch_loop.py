import torch
import torchvision

from data_loader import MyDataset
from model import RegressionModel
from functions import increase_saturation

from torch.utils.data import DataLoader
from IQA_pytorch import SSIM, LPIPSvgg, DISTS
from kornia.metrics import psnr

import numpy as np
import os

img_size = 512

# test dataset
test_input = '/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Test/Low'

# 텐서 데이터 불러오기
test_data = MyDataset(test_input, target_shape=(img_size, img_size), mode='val')

print("테스트 데이터셋 개수", test_data.len)

# 미니 배치 형태로 데이터 갖추기
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=16)

print("배치 개수",len(test_loader))

# CPU/GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

weights_list = []

folder_path = '/data/agc/agc_new/models'
filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# print(filenames)

for filename in filenames:
    w_number = filename.split('_')[-1].split('.')[0]
    weights_list.append(w_number)

total_score_threshold = 1

for weights in weights_list:
    print("agc_new_"+weights)

    # if weights in ["170", "381", "257", "331"]:
    #     pass

    # 모델 선언
    agc_new_model = RegressionModel().to(device) 
    MODEL_PATH = f"{folder_path}/agc_new_{weights}.pth"
    # MODEL_PATH = "/data/agc/agc_new/9_models_firm-glitter-104/agc_new_596.pth"
    model_name = MODEL_PATH.split('/')[-1].split('.')[0]
    agc_new_model.load_state_dict(torch.load(MODEL_PATH))

    # result_path = f"./outputs/{model_name}/"
    # if not os.path.exists(result_path):
    #     os.mkdir(result_path)

    n = len(test_loader) # 배치 개수

    ssim = SSIM()
    lpips = LPIPSvgg().to(device)
    dists = DISTS().to(device)

    ssim_list = []
    psnr_list = []
    lpips_list = []
    dists_list = []

    with torch.no_grad():
        agc_new_model.eval()
        # for inputs, gt in test_loader:
        #     inputs, gt = inputs.to(device), gt.to(device)
        for idx, imgs in enumerate(test_loader):
            inputs, gt = imgs[0].to(device), imgs[1].to(device)
            # inputs, gt, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
            # print(name)

            # Assuming `inputs`, `gt` is your normalized tensor with shape (B, C, H, W)
            # Get predictions from the network
            values = agc_new_model(inputs)

            # Instead of iterating over each image, perform batch operations
            bg = values[:,0].view(-1, 1, 1, 1) # torch.Size([8, 1, 1, 1])
            dg = values[:,1].view(-1, 1, 1, 1)
            saturation_factor = values[:,2]

            # Clamp the tensor
            saturation_factor = torch.clamp(saturation_factor, min=1e-4)
            saturation_factor = saturation_factor.view(-1, 1, 1, 1)

            # 0 =< i_map, r_i_map =< 1
            i_map, _ = torch.max(inputs, dim=1, keepdim=True)
            r_i_map = 1 - i_map

            # Gamma correction
            out_1 = inputs ** bg
            out_2 = inputs ** dg

            # Applying the transformations
            fu_1 = out_1 * r_i_map
            fu_2 = out_2 * i_map

            # Image addition is just an element-wise add
            result_imgs = fu_1 + fu_2

            # Constrain the values in a tensor to be 0~1
            result_imgs = torch.clamp(result_imgs, min=0, max=1)

            outputs = increase_saturation(result_imgs, saturation_factor)

            # torchvision.utils.save_image(outputs, result_path + str(name))
            # torchvision.utils.save_image(outputs, result_path + str(idx) + '.png')
            # torchvision.utils.save_image(high_img, result_gt_path + str(i) + '.png')
            ssim_value = ssim(outputs, gt, as_loss=False).item()
            # psnr_value = psnr(enhanced_img, high_img).item()
            val_psnr = psnr(outputs, gt, 1.0)
            val_lpips = lpips(outputs, gt, as_loss=False).item()

            val_dists = dists(outputs, gt, as_loss=False).item()
            
            ssim_list.append(ssim_value)
            psnr_list.append(val_psnr.cpu())
            lpips_list.append(val_lpips)
            dists_list.append(val_dists)

        SSIM_mean = np.mean(ssim_list)
        PSNR_mean = np.mean(psnr_list)
        LPIPS_mean = np.mean(lpips_list)
        DISTS_mean = np.mean(dists_list)

        total_score = SSIM_mean + PSNR_mean + (1-LPIPS_mean) + (1-DISTS_mean)

        # print('The SSIM Value is:', SSIM_mean)
        # print('The PSNR Value is:', PSNR_mean)
        # print('The LPIPS Value is:', LPIPS_mean)
        # if total_score > total_score_threshold or SSIM_mean >= 0.84 or PSNR_mean >= 21.8:
        f = open("/data/agc/agc_new/results.txt", 'a')
        f.write("agc_new_"+weights+'\n')
        f.write(f'The total score is:{total_score}\n')
        f.write(f'The SSIM Value is:{SSIM_mean}\n')
        f.write(f'The PSNR Value is:{PSNR_mean}\n')
        f.write(f'The LPIPS Value is:{LPIPS_mean}\n')
        f.write(f'The DISTS Value is:{DISTS_mean}\n')
        f.close()
        # if total_score > total_score_threshold:
            # total_score_threshold = total_score
    