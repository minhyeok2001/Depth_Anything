import torch
import torch.optim as optim
from tqdm import tqdm
from src.utils.check_device import device
import yaml
import wandb
from dotenv import load_dotenv,find_dotenv
from src.loss.loss_teacher import teacher_loss_function
import os
from src.models.model import DepthModel
from src.data.data_loader_teacher import dataloader

device = device

# hyper params
config_path = "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hyper_params = config["hyper_parameter"]
lr = hyper_params["learning_rate"]
num_epochs = hyper_params["epochs"]
patient = hyper_params["patient"]


# wandb setting
load_dotenv(find_dotenv())
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)
run = wandb.init(project="DepthAnything_teacher",entity="mhroh01-ajou-university",config=hyper_params)

model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False).to(device)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# dataloader

running_loss = 0.0
trial = 0

for epoch in range(num_epochs):
    prev_loss = running_loss
    running_loss = 0.0

    for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader)):
        # 입력 데이터도 device로 옮기기
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # outputs: (B, H, W)
        #print("inputs: ", inputs)
        #print("outputs: ", outputs)

        loss = teacher_loss_function(outputs, targets, disparity=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    wandb.log({"epoch_loss": epoch_loss, "epoch": epoch+1})

    if running_loss > prev_loss :
        if patient > trial:
            trial +=1
        else:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }
            torch.save(checkpoint, f'final_checkpoint.pth')
            print("Final checkpoint saved.")
            break
    else:
        trial = 0


run.finish()

print("TRAIN FINISHED")