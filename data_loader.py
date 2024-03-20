import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tr # 이미지 전처리 기능들을 제공하는 라이브러리
from torchvision.transforms.functional import InterpolationMode

import random
import torchvision.transforms.functional as TF



def tensor_transform(x, y):
    # Apply horizontal flip
    if random.random() > 0.5:
        x = TF.hflip(x)
        y = TF.hflip(y)

    # Apply vertical flip
    if random.random() > 0.5:
        x = TF.vflip(x)
        y = TF.vflip(y)

    # Add more transformations as needed

    return x, y

def get_paths_from_dir(folder_path):
    # List all files in the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # print(filenames) # a4645-Duggan_090426_7758.png
    paths = []
    # Initialize a zero array to store the images
    # images_array = np.zeros((len(filenames), target_shape[0], target_shape[1], 3))
    
    # for idx, filename in enumerate(filenames[:16]):
    #     image_path = os.path.join(folder_path, filename)
        
    #     # Read and resize the image
    #     img = cv2.imread(image_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     resized_img = cv2.resize(img, target_shape)
        
    #     # Add the resized image to the array
    #     images_array[idx] = resized_img
    # return images_array
    for filename in filenames:
        paths.append(os.path.join(folder_path, filename))
    print(folder_path)
    print("경로 길이",len(paths))
    return paths



def extract_numbers_from_folder(folder_path):
    # List all txt files in the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.txt')]
    len_file = len(filenames)
    
    numbers_list = []

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        
        # Read the file and extract numbers
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            numbers = list(map(float, line.split()))
            
            if len(numbers) != 2:
                raise ValueError(f"Expected 2 numbers in the file {filename}. Got {len(numbers)} numbers.")
            
            # Append the two numbers to the list
            numbers_list.append(numbers)

    # Convert the list to a numpy array of shape (1415, 2)
    numbers_array = np.array(numbers_list).reshape(len_file, 2)

    return numbers_array

# tranform 없이

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.x_data = self.x_data.permute(0,3,1,2)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class MyDataset(Dataset):

    def __init__(self, x_path, target_shape=None, mode='train'):
        self.x_path = get_paths_from_dir(x_path)
        # self.y_path = get_paths_from_dir('/data/agc/Illumination-Adaptive-Transformer/dataset/exposure/validation/Result_gt')
        # self.y_path = get_paths_from_dir('/data/agc/Illumination-Adaptive-Transformer/dataset/lol2/Test/Normal')
        self.y_path = get_paths_from_dir(x_path.replace('Low', 'Normal'))
        self.mode = mode
        # self.len = len(x_path)
        self.len = len(self.x_path)
        self.target_shape = target_shape

    def __getitem__(self, index):
        x_path = self.x_path[index]
        y_path = self.y_path[index]

        if self.mode == 'train':
            # Read and resize the image
            x = cv2.imread(x_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                x = cv2.resize(x, self.target_shape)
            inputs = torch.FloatTensor(x)
            inputs = inputs.permute(2,0,1)
            inputs = inputs/255.0

            y = cv2.imread(y_path)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                y = cv2.resize(y, self.target_shape)
            gt = torch.FloatTensor(y)
            gt = gt.permute(2,0,1)
            gt = gt/255.0

            inputs, gt = tensor_transform(inputs, gt)

            sample = inputs, gt

            filename = x_path.split('/')[-1]
            # print(filename) # a0609-_MG_3231.png

            # return inputs, gt, filename
            return sample
        
        if self.mode == 'val':
            # Read and resize the image
            x = cv2.imread(x_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                x = cv2.resize(x, self.target_shape)
            inputs = torch.FloatTensor(x)
            inputs = inputs.permute(2,0,1)
            inputs = inputs/255.0

            y = cv2.imread(y_path)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                y = cv2.resize(y, self.target_shape)
            gt = torch.FloatTensor(y)
            gt = gt.permute(2,0,1)
            gt = gt/255.0

            sample = inputs, gt

            filename = x_path.split('/')[-1]
            # print(filename) # a0609-_MG_3231.png

            # return inputs, gt, filename
            return sample
        
        elif self.mode == 'test':
            # Read and resize the image
            x = cv2.imread(x_path)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                x = cv2.resize(x, self.target_shape)
            inputs = torch.FloatTensor(x)
            inputs = inputs.permute(2,0,1)
            inputs = inputs/255.0

            filename = x_path.split('/')[-1]

            # return inputs, filename 
            return inputs

    def __len__(self):
        return self.len


# transform 하고

class MyTransform:

    def __call__(self, sample):
        inputs, gt = sample # sample 불러오기

        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2,0,1)
        
        gt = torch.FloatTensor(gt)
        gt = gt.permute(2,0,1)

        # 여기에 바로 .Compose() 넣기
        # transf = tr.Compose([tr.ToPILImage(), tr.ToTensor()])
        # transf = tr.Compose([tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transf = tr.Compose([
            tr.ToPILImage(),
            # tr.Resize(256, interpolation=InterpolationMode.BILINEAR),
            tr.Resize(224, interpolation=InterpolationMode.BILINEAR),
            # tr.CenterCrop(224),
            tr.ToTensor(),  # This will scale pixels to [0, 1]
            # tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        scaled_input = transf(inputs)
        scaled_gt = transf(gt)

        # print(scaled_input.max(), scaled_input.min())
        # print(scaled_gt.max(), scaled_gt.min())

        return scaled_input, scaled_gt


'''
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
tensor(-1.8044) tensor(-2.1179)
torch.Size([8, 3, 224, 224]) torch.Size([8, 3, 224, 224])
tensor(2.6400) tensor(-2.1179)
tensor(2.6400) tensor(-2.1179)
torch.Size([1, 3, 224, 224]) torch.Size([1, 3, 224, 224])
'''


'''
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1122.9642) tensor(-2.0357)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1127.4286) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.0357)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1136.3572) tensor(-2.1179)
tensor(1118.1956) tensor(-2.1179)
'''
'''
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(0.9961) tensor(0.0039)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
tensor(1.) tensor(0.)
'''

class MyTestDataset(Dataset):

    def __init__(self, x_data, transform=None):

        self.x_data = x_data # 넘파이 배열이 들어온다.
        self.transform = transform
        self.len = len(x_data)

    def __getitem__(self, index):
        sample = self.x_data[index]

        # self.transform이 None이 아니라면 전처리를 작업한다.
        if self.transform:
            sample = self.transform(sample)

        return sample # class TensorData(Dataset)과 다르게 넘파이 배열로 출력 되는 것에 유의 하도록 한다.

    def __len__(self):
        return self.len
    

class MyTestTransform:

    def __call__(self, sample):
        inputs = sample # sample 불러오기
        inputs = torch.FloatTensor(inputs)
        inputs = inputs.permute(2,0,1)

        # 여기에 바로 .Compose() 넣기
        transf = tr.Compose([
            tr.ToPILImage(),
            # tr.Resize(256, interpolation=InterpolationMode.BILINEAR),
            tr.Resize(224, interpolation=InterpolationMode.BILINEAR),
            # tr.CenterCrop(224),
            tr.ToTensor(),  # This will scale pixels to [0, 1]
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        final_output = transf(inputs)

        return final_output





class MyExposureDataset(Dataset):
    def __init__(self, input_dir, gt_dir, target_shape=None, mode='train'):
        self.mode = mode
        self.input_images = []
        self.gt_images = []
        # self.transform = transform
        self.target_shape = target_shape

        # Assuming the ground truth image name is part of the input image name
        for filename in os.listdir(gt_dir):
            for name in ['_0.JPG','_N1.5.JPG','_N1.JPG','_P1.5.JPG','_P1.JPG']:
                gt_image_path = os.path.join(gt_dir, filename)
                input_image_name = filename.split('.')[0] + name  # Modify this based on your naming pattern
                input_image_path = os.path.join(input_dir, input_image_name)

                self.input_images.append(input_image_path)
                self.gt_images.append(gt_image_path)

        self.len = len(self.input_images)

        # print(len(self.input_images))
        # print(self.input_images[5])
        # print(len(self.gt_images))
        # print(self.gt_images[5])

    def __getitem__(self, index):
        input_image = self.input_images[index] # Load input image from self.input_images[index]
        gt_image = self.gt_images[index] # Load ground truth image from self.gt_images[index]

        # if self.transform:
        #     input_image = self.transform(input_image)
        #     gt_image = self.transform(gt_image)

        if self.mode == 'train':
            # Read and resize the image
            x = cv2.imread(input_image)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                x = cv2.resize(x, self.target_shape)
            inputs = torch.FloatTensor(x)
            inputs = inputs.permute(2,0,1)
            inputs = inputs/255.0

            y = cv2.imread(gt_image)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                y = cv2.resize(y, self.target_shape)
            gt = torch.FloatTensor(y)
            gt = gt.permute(2,0,1)
            gt = gt/255.0

            inputs, gt = tensor_transform(inputs, gt)

            sample = inputs, gt

            return sample
        
        elif self.mode == 'val':
            # Read and resize the image
            x = cv2.imread(input_image)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                x = cv2.resize(x, self.target_shape)
            inputs = torch.FloatTensor(x)
            inputs = inputs.permute(2,0,1)
            inputs = inputs/255.0

            y = cv2.imread(gt_image)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)
            if self.target_shape:
                y = cv2.resize(y, self.target_shape)
            gt = torch.FloatTensor(y)
            gt = gt.permute(2,0,1)
            gt = gt/255.0

            sample = inputs, gt

            return input_image, gt_image
    
    def __len__(self):
        return self.len


def Read_resize_image(path, target_shape):
    x = cv2.imread(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if target_shape:
        if x.shape[0] > 512 or x.shape[1] > 512:
            x = cv2.resize(x, target_shape)
    inputs = torch.FloatTensor(x)
    inputs = inputs.permute(2,0,1)
    inputs = inputs/255.0

    filename = path.split('/')[-1]

    return inputs, filename


class ExposureValDataset(Dataset):
    # path_1 : GT path
    def __init__(self, input_path, path_1, target_shape=None):
        self.target_shape = target_shape
        self.input_path = input_path
        self.path_1 = path_1
        self.path_2 = path_1.replace('expert_a', 'expert_b')
        self.path_3 = path_1.replace('expert_a', 'expert_c')
        self.path_4 = path_1.replace('expert_a', 'expert_d')
        self.path_5 = path_1.replace('expert_a', 'expert_e')

        self.input_images = []
        self.gt_1_images = []
        self.gt_2_images = []
        self.gt_3_images = []
        self.gt_4_images = []
        self.gt_5_images = []
        # self.transform = transform

        # Assuming the ground truth image name is part of the input image name
        for filename in os.listdir(path_1):
            for name in ['_0.JPG','_N1.5.JPG','_N1.JPG','_P1.5.JPG','_P1.JPG']:
                gt_1_image_path = os.path.join(path_1, filename)
                gt_2_image_path = os.path.join(self.path_2, filename)
                gt_3_image_path = os.path.join(self.path_3, filename)
                gt_4_image_path = os.path.join(self.path_4, filename)
                gt_5_image_path = os.path.join(self.path_5, filename)
                input_image_name = filename.split('.')[0] + name  # Modify this based on your naming pattern
                input_image_path = os.path.join(self.input_path, input_image_name)

                self.input_images.append(input_image_path)
                self.gt_1_images.append(gt_1_image_path)
                self.gt_2_images.append(gt_2_image_path)
                self.gt_3_images.append(gt_3_image_path)
                self.gt_4_images.append(gt_4_image_path)
                self.gt_5_images.append(gt_5_image_path)

        self.len = len(self.input_images)
        

    def __getitem__(self, index):
        input_path = self.input_images[index]
        path_1 = self.gt_1_images[index]
        path_2 = self.gt_2_images[index]
        path_3 = self.gt_3_images[index]
        path_4 = self.gt_4_images[index]
        path_5 = self.gt_5_images[index]
        
        # Read and resize the image
        inputs, _ = Read_resize_image(input_path, self.target_shape)
        gt_1, _ = Read_resize_image(path_1, self.target_shape)
        gt_2, _ = Read_resize_image(path_2, self.target_shape)
        gt_3, _ = Read_resize_image(path_3, self.target_shape)
        gt_4, _ = Read_resize_image(path_4, self.target_shape)
        gt_5, _ = Read_resize_image(path_5, self.target_shape)
        
        sample = inputs, gt_1, gt_2, gt_3, gt_4, gt_5

        return sample
        
    def __len__(self):
        return self.len






class EvaluationDataset(Dataset):

    def __init__(self, path_1, path_2, path_3, path_4, path_5, gt_path, target_shape=(512,512)):
        self.path_1 = get_paths_from_dir(path_1)
        self.path_2 = get_paths_from_dir(path_2)
        self.path_3 = get_paths_from_dir(path_3)
        self.path_4 = get_paths_from_dir(path_4)
        self.path_5 = get_paths_from_dir(path_5)
        self.gt_path = get_paths_from_dir(gt_path)
        self.len = len(self.path_1)
        self.target_shape = target_shape

    def __getitem__(self, index):
        path_1 = self.path_1[index]
        path_2 = self.path_2[index]
        path_3 = self.path_3[index]
        path_4 = self.path_4[index]
        path_5 = self.path_5[index]
        gt_path = self.gt_path[index]

        # Read and resize the image
        input_1, filename_1 = Read_resize_image(path_1)
        input_2, _ = Read_resize_image(path_2)
        input_3, _ = Read_resize_image(path_3)
        input_4, _ = Read_resize_image(path_4)
        input_5, _ = Read_resize_image(path_5)
        gt, _ = Read_resize_image(gt_path)

        sample = filename_1, input_1, input_2, input_3, input_4, input_5, gt


        return sample
        

    def __len__(self):
        return self.len


