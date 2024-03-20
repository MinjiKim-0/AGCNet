import torch

def increase_saturation(rgb_images, saturation_factor):
    # Assuming rgb_images is of shape [B, C, H, W] and in the range [0, 1]
    
    # Find the maximum across the color channels
    max_rgb, _ = torch.max(rgb_images, dim=1, keepdim=True)
    
    # Calculate the difference between the max and the other channels
    diff = max_rgb - rgb_images
    # diff shape torch.Size([16, 3, 224, 224])
    
    # Scale down the difference by the saturation factor
    # If saturation_factor is 1, there is no change
    # If saturation_factor is < 1, the colors will become more saturated
    # Note that saturation_factor should not be more than 1 as it would desaturate the image
    new_diff = diff / saturation_factor
    
    # Subtract the scaled difference from the max value to get the new RGB values
    saturated_rgb = max_rgb - new_diff
    
    # Clamp the values to maintain them in the [0, 1] range
    saturated_rgb = torch.clamp(saturated_rgb, min=0, max=1)
    
    return saturated_rgb