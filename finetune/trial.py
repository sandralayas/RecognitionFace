import sys
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from AgeEstimator import AgeEstimator
from fgnet_dataset import FGNETDataset, transform 

def main():
    # --- Step 1: Model Setup (unchanged from previous steps) ---
    # Add the backbones directory to the system path
    backbones_dir = os.path.join(r'C:\Users\sandr\Documents\git\insightface\recognition\arcface_torch\backbones')
    sys.path.append(backbones_dir)

    # Import the iresnet module
    import iresnet

    # Define the full path to your downloaded 'backbone.pth' file
    model_path = os.path.join(r'C:\Users\sandr\Documents\git\insightface\models\r100-arcface-emore', 'backbone.pth')
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: The model file was not found at '{model_path}'")
        sys.exit()

    # Load the pre-trained backbone architecture
    backbone = iresnet.iresnet100(pretrained=False)
    
    # Load the state dictionary with map_location and strict=False
    backbone.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    # Create an instance of our new AgeEstimator model
    model = AgeEstimator(backbone)
    
    print("Full model (backbone + head) loaded successfully!")
    print(model)

    # --- Step 2: Data Loading ---
    # IMPORTANT: Update this path to your actual FG-NET dataset location
    fgnet_root_dir = r"C:\Users\sandr\Documents\git\insightface\finetuning\FGNET\ImageFolders"
    print(f"Checking for images in: {fgnet_root_dir}")

    # Create the dataset and dataloader
    try:
        fgnet_dataset = FGNETDataset(root_dir=fgnet_root_dir, transform=transform)
        train_loader = DataLoader(fgnet_dataset, batch_size=32, shuffle=True, num_workers=4)
        print(f"Found {len(fgnet_dataset)} images in the dataset.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your 'fgnet_root_dir' is correct and the folder structure is valid.")
        sys.exit()

    # --- Step 3: Training Loop Setup ---
    # Define the Loss Function and Optimizer
    criterion = nn.L1Loss() 
    
    # IMPORTANT: Only optimize the parameters of the new 'head' layers.
    # The backbone parameters are frozen and will not be updated.
    optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Starting training on device: {device}")

    # --- Step 4: Training and Evaluation Loop ---
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0

        for i, (images, ages) in enumerate(train_loader):
            images = images.to(device)
            ages = ages.float().to(device) 

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training MAE: {avg_loss:.4f}")

    # --- Step 5: Save the Fine-Tuned Model ---
    save_path = 'age_estimator_finetuned.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()