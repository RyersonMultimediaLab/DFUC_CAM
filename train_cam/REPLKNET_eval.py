import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from sklearn.decomposition import PCA
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import warnings
import csv
warnings.filterwarnings('ignore')

project_path = "/project/path/CAM_Back_Again"
models_dir = '/models/dir'
validation_dir = '/validation/images'
validation_mask_dir = '/validation/mask'
new_heatmap_dir = '/heatmap/output'

os.makedirs(new_heatmap_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model configuration
model_config = {
    "class_n": 2,
    "unit_n": 1024,
    "input_size": 384,
    "size": 12,
    "lr": 1e-05,
    "weight_decay": 0.0005,
    "channels": [128, 256, 512, 1024]
}

def load_model(model_path):
    if 'convnext' in model_path.lower():
        model = timm.create_model("convnext_base_384_in22ft1k", pretrained=False, num_classes=model_config["class_n"])
    elif 'replknet' in model_path.lower():
        model_config["model_name"] = "RepLKNet-31B"
        model_config["channels"] = [128,256,512,1024]
        model = build_model(model_config)
    else:
        raise ValueError(f"Unsupported model type: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device)

def get_feature_maps(x, model):
    features = []
    def hook_fn(module, input, output):
        features.append(output.detach())

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            handle = module.register_forward_hook(hook_fn)
            break

    _ = model(x)
    handle.remove()

    return features[0]

def get_cam_heatmap(features, model, target_class):
    b, c, h, w = features.shape
    features = features.reshape(c, h*w)
    weights = model.head.weight.data[target_class].to(device)
    heatmap = torch.mm(weights.unsqueeze(0), features).reshape(h, w)
    heatmap = F.relu(heatmap)
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (model_config["input_size"], model_config["input_size"]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

def get_pc1_heatmap(features):
    b, c, h, w = features.shape
    features = features.cpu().numpy().reshape(c, h*w).T
    pca = PCA(n_components=1)
    heatmap = pca.fit_transform(features).reshape(h, w)
    heatmap = np.abs(heatmap)
    heatmap = cv2.resize(heatmap, (model_config["input_size"], model_config["input_size"]))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

def load_and_preprocess_validation_data(validation_dir, validation_mask_dir):
    validation_images = []
    validation_masks = []
    for img_path in glob.glob(os.path.join(validation_dir, '*.png')):
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(validation_mask_dir, img_name)

        if os.path.exists(mask_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (model_config["input_size"], model_config["input_size"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1) / 255.0 

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (model_config["input_size"], model_config["input_size"]))
            mask = (mask > 0).astype(np.float32)

            validation_images.append(img)
            validation_masks.append(mask)

    return np.array(validation_images), np.array(validation_masks)

def dice_score(pred, target):
    smooth = 1e-5
    intersection = np.sum(pred * target)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def iou_score(pred, target):
    smooth = 1e-5
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_heatmap(heatmap, ground_truth, threshold=0.5):
    binary_heatmap = (heatmap > threshold).astype(np.float32)
    dilated_heatmap = binary_dilation(binary_heatmap, iterations=5)  

    dice = dice_score(dilated_heatmap, ground_truth)
    iou = iou_score(dilated_heatmap, ground_truth)

    return dice, iou
def generate_and_evaluate_heatmaps(model, validation_images, validation_masks, localization_method, model_filename):
    model.eval()
    dice_scores = []
    iou_scores = []

    fig, axs = plt.subplots(10, 3, figsize=(15, 50))
    fig.suptitle(f"{model_filename} - {localization_method}", fontsize=16)

    for idx, (img, mask) in enumerate(zip(validation_images[:10], validation_masks[:10])):
        x = torch.from_numpy(img).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logit = model(x)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        _, idx_sort = h_x.sort(0, True)
        target_class = idx_sort[0].item()

        features = get_feature_maps(x, model)

        if localization_method.lower() == "cam":
            heatmap = get_cam_heatmap(features, model, target_class)
        elif localization_method.lower() == "pc1":
            heatmap = get_pc1_heatmap(features)
        else:
            raise ValueError(f"Unsupported localization method: {localization_method}")

        dice, iou = evaluate_heatmap(heatmap, mask)
        dice_scores.append(dice)
        iou_scores.append(iou)

       
        axs[idx, 0].imshow(img.transpose(1, 2, 0))
        axs[idx, 0].set_title(f'Original Image {idx+1}')
        axs[idx, 0].axis('off')

        axs[idx, 1].imshow(heatmap, cmap='jet')
        axs[idx, 1].set_title(f'{localization_method} Heatmap')
        axs[idx, 1].axis('off')

        axs[idx, 2].imshow(img.transpose(1, 2, 0))
        axs[idx, 2].imshow(heatmap, cmap='jet', alpha=0.5)
        axs[idx, 2].set_title('Overlay')
        axs[idx, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    
    model_method_dir = os.path.join(new_heatmap_dir, f"{model_filename}_{localization_method}")
    os.makedirs(model_method_dir, exist_ok=True)

    
    plt.savefig(os.path.join(model_method_dir, 'heatmaps.png'))
    plt.close()

    return np.mean(dice_scores), np.mean(iou_scores)

print("Loading validation data...")
validation_images, validation_masks = load_and_preprocess_validation_data(validation_dir, validation_mask_dir)
print(f"Loaded {len(validation_images)} validation images and masks.")
validation_images = validation_images[:10]
validation_masks = validation_masks[:10]

model_files = glob.glob(os.path.join(models_dir, "*.pth"))
results = []

print(f"\nFound {len(model_files)} model files:")
for model_file in model_files:
    print(f"  - {os.path.basename(model_file)}")

for model_file in model_files:
    model_filename = os.path.basename(model_file)  # Keep full filename
    print(f"\nProcessing model: {model_filename}")

    try:
        print(f"  Loading model from file: {model_file}")
        model = load_model(model_file)
        model.to(device)
        print(f"  Model loaded successfully and moved to {device}")
    except Exception as e:
        print(f"  Error loading model {model_filename}: {str(e)}")
        continue

    for method in ["CAM", "PC1"]:
        print(f"  Generating and evaluating {method} heatmaps for {model_filename}...")
        mean_dice, mean_iou = generate_and_evaluate_heatmaps(model, validation_images, validation_masks, method, model_filename)
        results.append({
            'model': model_filename,
            'method': method,
            'mean_dice': mean_dice,
            'mean_iou': mean_iou
        })
        print(f"    {model_filename} - {method}: Mean Dice = {mean_dice:.4f}, Mean IoU = {mean_iou:.4f}")

results.sort(key=lambda x: x['mean_dice'], reverse=True)
print("\nFinal Results (sorted by Mean Dice):")
for result in results:
    print(f"{result['model']} - {result['method']}: Mean Dice = {result['mean_dice']:.4f}, Mean IoU = {result['mean_iou']:.4f}")

csv_filename = os.path.join(new_heatmap_dir, 'heatmap_evaluation_results.csv')
with open(csv_filename, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['model', 'method', 'mean_dice', 'mean_iou'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"\nEvaluation completed and results saved to {csv_filename}")
print(f"Total models evaluated: {len(model_files)}")
print(f"Total evaluation entries: {len(results)}")
print(f"Heatmap plots are stored in: {new_heatmap_dir}")