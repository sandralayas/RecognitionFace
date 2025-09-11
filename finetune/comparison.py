import sys
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from AgeEstimator import AgeEstimator
from fgnet_dataset import FGNETDataset, transform 

# --- Setup (Same as before) ---
# Add the backbones directory to the system path
backbones_dir = os.path.join('iresnet50.pth')
sys.path.append(backbones_dir)
import iresnet

# Define your specific file paths
backbone_model_path = 'iresnet50.pth'
finetuned_model_path = 'gpu_finetuned_1.pth'

# IMPORTANT: Update this path to your actual FG-NET dataset location
fgnet_root_dir = "FGNETvalidation"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation Function ---
def evaluate_model_mae(model, data_loader, device):
    """
    Calculates the Mean Absolute Error (MAE) of a model on a dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    num_samples = 0
    criterion = nn.L1Loss(reduction='sum')

    with torch.no_grad():
        for images, ages in data_loader:
            images = images.to(device)
            ages = ages.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, ages)
            
            total_loss += loss.item()
            num_samples += images.size(0)
    
    mae = total_loss / num_samples
    return mae

# --- Model Loading and Comparison ---
if __name__ == "__main__":
    
    # 1. Load the pre-trained backbone model
    if not os.path.exists(backbone_model_path):
        print(f"Error: Backbone model file not found at '{backbone_model_path}'")
        sys.exit()
    backbone = iresnet.iresnet100(pretrained=False)
    backbone.load_state_dict(torch.load(backbone_model_path, map_location=device), strict=False)

    # 2. Instantiate the "Without Fine-tuning" model (our baseline)
    # The head layers are randomly initialized, as they haven't been trained yet.
    model_without_finetuning = AgeEstimator(backbone).to(device)
    print("Baseline model (without fine-tuning) created.")

    # 3. Instantiate and load the "After Fine-tuning" model
    if not os.path.exists(finetuned_model_path):
        print(f"Error: Fine-tuned model not found at '{finetuned_model_path}'")
        print("Please run the fine-tuning script first to create this file.")
        sys.exit()
    
    # Create a new model instance and load the entire state dictionary.
    # This assumes your finetuning script saved the full model state.
    model_finetuned = AgeEstimator(backbone).to(device)
    model_finetuned.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    print("Fine-tuned model loaded.")

    # 4. Load the validation dataset
    try:
        fgnet_dataset = FGNETDataset(root_dir=fgnet_root_dir, transform=transform)
        # Use a small batch size for evaluation
        val_loader = DataLoader(fgnet_dataset, batch_size=32, shuffle=False, num_workers=4) 
        print(f"Loaded {len(fgnet_dataset)} images for validation.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit()
    
    # 5. Perform the evaluation and comparison
    print("\nStarting comparison...")

    # Evaluate the baseline model
    mae_without_finetuning = evaluate_model_mae(model_without_finetuning, val_loader, device)
    print(f"MAE without fine-tuning: {mae_without_finetuning:.2f} years")

    # Evaluate the fine-tuned model
    mae_finetuned = evaluate_model_mae(model_finetuned, val_loader, device)
    print(f"MAE with fine-tuning: {mae_finetuned:.2f} years")
    
    print("\nComparison complete.")
    if mae_finetuned < mae_without_finetuning:
        print("Success: Fine-tuning significantly improved the model's performance!")
    else:
        print("Note: Fine-tuning did not improve the model's performance in this test.")