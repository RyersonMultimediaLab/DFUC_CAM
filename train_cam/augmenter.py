import os
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm
import json
import shutil
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def apply_realistic_augmentations(images, target_size=(224, 224)):
    try:
        # Resize and convert to tensor, then move to GPU if available
        images = torch.stack([T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])(img) for img in images]).to(device)
        
        # 1. Motion blur (simulating hand shake)
        motion_blur_prob = 0.3
        motion_blur_mask = torch.rand(images.size(0)) < motion_blur_prob
        if motion_blur_mask.any():
            kernel_sizes = [random.choice([3, 5, 7]) for _ in range(images[motion_blur_mask].size(0))]
            angles = torch.rand(images[motion_blur_mask].size(0)) * 360
            for i, (size, angle) in enumerate(zip(kernel_sizes, angles)):
                kernel = torch.zeros((size, size)).to(device)
                kernel[size // 2, :] = 1
                kernel = TF.rotate(kernel.unsqueeze(0), angle.item())
                kernel = kernel / kernel.sum()
                images[motion_blur_mask][i] = TF.gaussian_blur(images[motion_blur_mask][i], kernel_size=size, sigma=(0.1, 2.0))
        
        # 2. Defocus blur (simulating out-of-focus shots)
        defocus_blur_prob = 0.2
        defocus_blur_mask = torch.rand(images.size(0)) < defocus_blur_prob
        if defocus_blur_mask.any():
            radii = torch.randint(1, 3, (images[defocus_blur_mask].size(0),)).to(device)  # radius range: 1 to 2
            for i, radius in enumerate(radii):
                kernel_size = int(2 * radius.item() + 1)
                images[defocus_blur_mask][i] = TF.gaussian_blur(images[defocus_blur_mask][i], kernel_size=kernel_size, sigma=(0.1, 2.0))
        
        # 3. Exposure variation (under/overexposure)
        exposure_prob = 0.4
        exposure_mask = torch.rand(images.size(0)) < exposure_prob
        exposure_factors = torch.rand(images[exposure_mask].size(0)).to(device) * 1.5 + 0.5  # range: 0.5 to 2.0
        images[exposure_mask] = torch.clamp(images[exposure_mask] * exposure_factors.view(-1, 1, 1, 1), 0, 1)
        
        # 4. Color temperature variation (warm/cool lighting)
        color_temp_prob = 0.3
        color_temp_mask = torch.rand(images.size(0)) < color_temp_prob
        temp_factors = torch.rand(images[color_temp_mask].size(0)).to(device) * 0.4 + 0.8  # range: 0.8 to 1.2
        images[color_temp_mask][:, 0, :, :] *= temp_factors.view(-1, 1, 1)  # Adjust red channel
        images[color_temp_mask][:, 2, :, :] *= (2 - temp_factors).view(-1, 1, 1)  # Adjust blue channel inversely
        
        # 5. Shadow effects (uneven lighting)
        shadow_prob = 0.2
        shadow_mask = torch.rand(images.size(0)) < shadow_prob
        if shadow_mask.any():
            h, w = images.shape[2:]
            shadow_center = torch.rand(images[shadow_mask].size(0), 2).to(device)
            x, y = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
            for i, center in enumerate(shadow_center):
                distance = torch.sqrt((x - center[0] * h)**2 + (y - center[1] * w)**2)
                shadow = torch.clamp(distance / (0.5 * torch.sqrt(torch.tensor(h**2 + w**2).to(device))), 0.7, 1)
                images[shadow_mask][i] *= shadow
        
        # 6. Slight rotations (camera not perfectly aligned)
        rotation_prob = 0.5
        rotation_mask = torch.rand(images.size(0)) < rotation_prob
        if rotation_mask.any():
            angles = (torch.rand(images[rotation_mask].size(0)) - 0.5) * 20  # -10 to 10 degrees
            rotated_images = []
            for img, angle in zip(images[rotation_mask], angles):
                rotated_images.append(TF.rotate(img, angle.item()))
            images[rotation_mask] = torch.stack(rotated_images)
        
        # 7. Perspective distortion (camera not perfectly perpendicular)
        perspective_prob = 0.3
        perspective_mask = torch.rand(images.size(0)) < perspective_prob
        if perspective_mask.any():
            h, w = images.shape[2:]
            for i in range(images[perspective_mask].size(0)):
                startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
                endpoints = [[random.randint(0, w // 10), random.randint(0, h // 10)],
                             [random.randint(w - w // 10, w), random.randint(0, h // 10)],
                             [random.randint(w - w // 10, w), random.randint(h - h // 10, h)],
                             [random.randint(0, w // 10), random.randint(h - h // 10, h)]]
                images[perspective_mask][i] = TF.perspective(images[perspective_mask][i], startpoints, endpoints)
        
        return torch.clamp(images, 0, 1)
    except Exception as e:
        print(f"Error in apply_realistic_augmentations: {str(e)}")
        print(traceback.format_exc())
        return None

def process_ulcer_dataset(input_folder, output_folder, num_augmentations=9, batch_size=32, checkpoint_interval=100, target_size=(224, 224)):
    checkpoint_file = 'augmentation_checkpoint.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {'split': 'train', 'file_index': 0}

    for split in ['train', 'val']:
        if split < checkpoint['split']:
            continue

        input_split_folder = os.path.join(input_folder, split)
        output_split_folder = os.path.join(output_folder, split)
        os.makedirs(output_split_folder, exist_ok=True)
        
        all_files = []
        for root, _, files in os.walk(input_split_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_files.append((root, file))

        for i in tqdm(range(checkpoint['file_index'], len(all_files), batch_size), desc=f"Processing {split} set"):
            batch_files = all_files[i:i+batch_size]
            batch_images = []
            batch_paths = []
            
            for root, file in batch_files:
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_split_folder)
                output_subfolder = os.path.join(output_split_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Save the original image
                original_output_path = os.path.join(output_subfolder, file)
                shutil.copy2(input_path, original_output_path)
                
                try:
                    image = Image.open(input_path).convert('RGB')
                    batch_images.append(image)
                    batch_paths.append((output_subfolder, file))
                except Exception as e:
                    print(f"Error processing image {input_path}: {str(e)}")
            
            # Process the batch for augmentations
            augmented_images = apply_realistic_augmentations(batch_images * num_augmentations, target_size)
            
            if augmented_images is None:
                print(f"Error: augmented_images is None for batch starting at index {i}")
                continue
            
            # Save the augmented images
            for j, (output_subfolder, file) in enumerate(batch_paths):
                for k in range(num_augmentations):
                    idx = j * num_augmentations + k
                    output_path = os.path.join(output_subfolder, f"{os.path.splitext(file)[0]}_aug_{k+1}.png")
                    try:
                        T.ToPILImage()(augmented_images[idx].cpu()).save(output_path)
                    except Exception as e:
                        print(f"Error saving augmented image: {str(e)}")
                        print(f"augmented_images shape: {augmented_images.shape}")
                        print(f"idx: {idx}")
            
            # Update and save checkpoint
            checkpoint['file_index'] = i + batch_size
            if (i + batch_size) % checkpoint_interval == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
        
        # Reset file index for next split
        checkpoint['split'] = 'val' if split == 'train' else 'completed'
        checkpoint['file_index'] = 0
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

    print("Ulcer dataset augmentation completed! Dataset is now 10 times larger.")
    os.remove(checkpoint_file)

if __name__ == "__main__":
    input_folder = "/path/input_folder"
    output_folder = "/path/output_folder"
    num_augmentations = 9  # 9 augmented images + 1 original = 10x dataset size
    batch_size = 32  # Adjust based on your GPU memory
    checkpoint_interval = 100  # Save checkpoint every 100 batches
    target_size = (224, 224)  # Target size for all images
    
    process_ulcer_dataset(input_folder, output_folder, num_augmentations, batch_size, checkpoint_interval, target_size)
