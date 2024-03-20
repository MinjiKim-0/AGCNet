import torch
from data_loader import MyDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import trange # for문의 진행 상황을 bar 형태로 출력

from model import RegressionModel
from kornia.metrics import psnr
from IQA_pytorch import SSIM, LPIPSvgg, NLPD, MS_SSIM, DISTS
from functions import increase_saturation

import wandb
import logging


learning_rate = 3e-5 # 'CosineAnnealingLR' # 1e-5 # or 1e-4 (1e-2 : too high) 
num_epochs = 2000
train_batch_size = 16
val_batch_size = 16
img_size = 512

# (Initialize logging)
experiment = wandb.init(project="ACGNet2-ResNet50")
experiment.config.update(
    dict(epochs=num_epochs, batch_size=train_batch_size, learning_rate=learning_rate))


logging.info(f'''Starting training:
    Epochs:          {num_epochs}
    Batch size:      {train_batch_size}
    Learning rate:   {learning_rate}
''')


train_input ='/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Train/Low'
# train_gt = '/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Train/Normal'
val_input = '/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Test/Low'
# val_gt = '/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Test/Normal'


# 텐서 데이터 불러오기
train_data = MyDataset(train_input, target_shape=(img_size, img_size))
val_data = MyDataset(val_input, target_shape=(img_size, img_size), mode='val') # , target_shape=(val_img_size, val_img_size)

print("train 데이터셋 개수", train_data.len)
print("val 데이터셋 개수", val_data.len)

# 미니 배치 형태로 데이터 갖추기
train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=16) 
val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=16) 

print("train 배치 개수",len(train_loader))
print("val 배치 개수",len(val_loader))


'''
train_dataiter = iter(train_loader)
t_images, t_labels = next(train_dataiter)
print(t_images.size(), t_labels.size())

val_dataiter = iter(val_loader)
v_images, v_labels = next(val_dataiter)
print(v_images.size(), v_labels.size())
'''

# CPU/GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

# 모델 선언
agc_new_model = RegressionModel().to(device) 
print(agc_new_model)


# 모델 학습하기
L1Loss = nn.L1Loss()
# criterion = nn.MSELoss() # nn에서 제공하는 mean squared error
# ms_ssim = MS_SSIM().to(device)
nlpd = NLPD(k=4).to(device)
lpips = LPIPSvgg().to(device)
dists = DISTS().to(device)

optimizer = optim.AdamW(agc_new_model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
# optimizer = optim.Adam(agc_new_model.parameters(), lr=learning_rate, weight_decay=1e-7)
# optimizer = optim.SGD(agc_new_model.parameters(), lr=0.1, momentum=0.9)
# optimizer = optim.AdamW(agc_new_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7, amsgrad=False)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
# scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, 
#                                                                  T_mult=1, eta_min=0.0000001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

ssim = SSIM()


def validation_loss(dataloader):
    running_loss = 0.0
    running_val_psnr = 0.0
    running_val_ssim = 0.0
    running_val_lpips = 0.0
    running_val_dists = 0.0

    n = len(dataloader)

    with torch.no_grad():
        agc_new_model.eval()
        for imgs in dataloader:
            # val_input, val_gt = data[0].to(device), data[1].to(device) # validation input과 gt
            # output_value = agc_new_model(val_input)
            
            # # Print outputs and labels to debug
            # print("Outputs:", outputs)
            # print("Labels:", labels)

            # output_img = 

            inputs, gt = imgs[0].to(device), imgs[1].to(device)

            # Get predictions from the network
            values = agc_new_model(inputs)

            # `inputs`, `gt` 범위 : tensor(2.6400) tensor(-2.1179)
            # Assuming `inputs`, `gt` is your normalized tensor with shape (B, C, H, W)
            # denormalized_inputs = denormalize(inputs)
            # denormalized_gt = denormalize(gt)

            # Instead of iterating over each image, perform batch operations
            bg = values[:,0].view(-1, 1, 1, 1) # torch.Size([8, 1, 1, 1])
            dg = values[:,1].view(-1, 1, 1, 1)
            # s = values[:,2].view(-1, 1, 1, 1)
            # s = values[:,2].view(-1, 1, 1)
            # print(s.shape)
            saturation_factor = values[:,2]
            # print("saturation_factor",saturation_factor)
            # Clamp the tensor
            saturation_factor = torch.clamp(saturation_factor, min=1e-4)
            # print("max_saturation_factor",saturation_factor)
            saturation_factor = saturation_factor.view(-1, 1, 1, 1)
            # print("saturation_factor_shape",saturation_factor.shape)

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
            # denormalized_gt = torch.clamp(denormalized_gt, min=0, max=1)

            outputs = increase_saturation(result_imgs, saturation_factor)
            
            # loss = criterion(outputs, gt)
            # loss = MS_SSIM_L1_Loss(outputs, gt)+0.04*loss_network(outputs, gt)val_lpips = lpips(outputs, gt, as_loss=False)
            # loss = 0.35*L1Loss(outputs, gt)+0.3*ms_ssim(outputs, gt).mean()+0.3*nlpd(outputs, gt).mean()+0.05*lpips(outputs, gt).mean()
            loss = L1Loss(outputs, gt)+0.8*nlpd(outputs, gt).mean()+0.1*lpips(outputs, gt).mean()
            running_loss += loss.item()

            val_psnr = psnr(outputs, gt, 1.0)
            running_val_psnr += val_psnr.item()

            # val_ssim = ssim(outputs, gt, 5)
            val_ssim = ssim(outputs, gt, as_loss=False)
            running_val_ssim += val_ssim.mean().item()

            val_lpips = lpips(outputs, gt, as_loss=False)
            running_val_lpips += val_lpips.mean().item()

            val_dists = dists(outputs, gt, as_loss=False)
            running_val_dists += val_dists.mean().item()

            # loss = criterion(output_img, val_gt)
            # running_loss += loss.item()
   
    return running_loss/n, running_val_psnr/n, running_val_ssim/n, running_val_lpips/n, running_val_dists/n


train_loss_list = [] # 그래프를 그리기 위한 loss 저장용 리스트 
val_loss_list = []
n = len(train_loader) # 배치 개수

threshold_psnr = 18 # 임의의 psnr 임계값
threshold_ssim = 0.80 # 임의의 ssim 임계값
threshold_lpips = 0.2 # 임의의 lpips 임계값
threshold_dists = 0.1 # 임의의 dists 임계값
threshold_loss = 1 # 임의의 loss 임계값

pbar = trange(num_epochs)
global_step = 0

torch.autograd.set_detect_anomaly(True)
for epoch in pbar:
    agc_new_model.train()
    running_loss = 0.0
    for imgs in train_loader:
    # for inputs, gt in train_loader:
        inputs, gt = imgs[0].to(device), imgs[1].to(device)
        # inputs, gt = inputs.to(device), gt.to(device)
        # print(inputs[0])

        optimizer.zero_grad()

        # Get predictions from the network
        values = agc_new_model(inputs)
        # print(values)

        # `inputs`, `gt` 범위 : tensor(2.6400) tensor(-2.1179)
        # Assuming `inputs`, `gt` is your normalized tensor with shape (B, C, H, W)
        # denormalized_inputs = denormalize(inputs)
        # denormalized_gt = denormalize(gt)

        # Instead of iterating over each image, perform batch operations
        bg = values[:,0].view(-1, 1, 1, 1) # torch.Size([8, 1, 1, 1])
        dg = values[:,1].view(-1, 1, 1, 1)
        # s = values[:,2].view(-1, 1, 1, 1)
        saturation_factor = values[:,2]
        # print("saturation_factor",saturation_factor)
        # Clamp the tensor
        saturation_factor = torch.clamp(saturation_factor, min=1e-4)
        # print("max_saturation_factor",saturation_factor)
        saturation_factor = saturation_factor.view(-1, 1, 1, 1)
        # print("saturation_factor_shape",saturation_factor.shape)
        
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
        # denormalized_gt = torch.clamp(denormalized_gt, min=0, max=1)

        outputs = increase_saturation(result_imgs, saturation_factor)

        # Rescale the images back to [0, 255] range
        # outputs = outputs * 255.0
        # gt = denormalized_gt * 255.0
        
        # loss = criterion(outputs, gt)
        # loss = 0.35*L1Loss(outputs, gt)+0.3*ms_ssim(outputs, gt).mean()+0.3*nlpd(outputs, gt).mean()+0.05*lpips(outputs, gt).mean()
        loss = L1Loss(outputs, gt)+0.8*nlpd(outputs, gt).mean()+0.1*lpips(outputs, gt).mean()
        # print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        '''
        # Post-processing for visualization or other non-gradient purposes
        with torch.no_grad():
            # Convert the image from RGB to HSV color space
            hsv_img = rgb_to_hsv(result_imgs.cpu())  # Move to CPU to avoid GPU memory usage

            # Correct Saturation (non-differentiable operation)
            hsv_img[:, 1, :, :] = torch.pow(hsv_img[:, 1, :, :], s.cpu())

            # Convert the image from HSV to RGB color space
            rgb_img = hsv_to_rgb(hsv_img)

            # Constrain the values in a tensor to be 0~1
            rgb_img = torch.clamp(rgb_img, min=0, max=1)
        '''

    # scheduler.step()
    global_step += 1

    train_loss = running_loss / n
    # train_loss_list.append(train_loss)    
    val_loss, val_psnr, val_ssim, val_lpips, val_dists = validation_loss(val_loader)
    # val_loss_list.append(val_loss)
    
    logging.info('Validation PSNR, SSIM score: {}, {}'.format(val_psnr, val_ssim))
    
    inputs_dis = inputs[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    i_map_dis = i_map[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    r_i_map_dis = r_i_map[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    out_1_dis = out_1[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    out_2_dis = out_2[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    result_imgs_dis = result_imgs[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    outputs_dis = outputs[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    gt_dis = gt[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    # log metrics to wandb
    experiment.log({'train loss': train_loss,
                    'val loss' : val_loss,
                    'step': global_step,
                    'epoch': epoch,
                    'validation PSNR': val_psnr,
                    'validation SSIM': val_ssim,
                    'validation LPIPS': val_lpips,
                    'validation DISTS': val_dists,
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'input': wandb.Image(inputs_dis),
                    'i_map': wandb.Image(i_map_dis),
                    'r_i_map': wandb.Image(r_i_map_dis),
                    'out_1': wandb.Image(out_1_dis),
                    'out_2': wandb.Image(out_2_dis),
                    'g_corrected': wandb.Image(result_imgs_dis),
                    'target': {
                        's_corrected': wandb.Image(outputs_dis),
                        'true': wandb.Image(gt_dis)
                    }
                    })
    
    pbar.set_postfix({'epoch': epoch + 1, 'train loss' : train_loss, 'validation loss' : val_loss, 'val_psnr' : val_psnr, "val_ssim" : val_ssim, "val_lpips" : val_lpips, "val_dists" : val_dists})
    


    # 저장하기
    PATH = f'./models/agc_new_{epoch}.pth' # 모델 저장 경로 
    # if val_loss < threshold_loss:
    if val_psnr > threshold_psnr or val_ssim > threshold_ssim or val_lpips < threshold_lpips or val_dists < threshold_dists or val_loss < threshold_loss:
        torch.save(agc_new_model.state_dict(), PATH)
        logging.info(f'Checkpoint {epoch} saved!')
        # threshold_loss = val_loss
        if val_psnr > threshold_psnr:
            threshold_psnr = val_psnr
        elif val_ssim > threshold_ssim:
            threshold_ssim = val_ssim
        elif val_lpips < threshold_lpips:
            threshold_lpips = val_lpips
        elif val_dists < threshold_dists:
            threshold_dists = val_dists
        elif val_loss < threshold_loss:
            threshold_loss = val_loss


### 학습이 끝난 후 세션 종료를 위한 .finish() 호출 ###
wandb.finish()