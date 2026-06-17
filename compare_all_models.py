from models import *
from dataset import *
from evaluation import *
from plotting import *
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




if __name__ == "__main__":
    fixed_batch_size = 8
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=fixed_batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = ContrastiveLoss()

    koch_model = SiameseNetwork()
    koch_weights = "Koch/weights_SEED=42.pth"
    koch_model.load_state_dict(torch.load(koch_weights))
    koch_model = koch_model.to(device)

    resnet_model = SiameseNetworkResnet(weights=None)
    resnet_weights = "ResNet18/weights_SEED=42.pth"
    resnet_model.load_state_dict(torch.load(resnet_weights))
    resnet_model = resnet_model.to(device)

    pretrained_resnet_model = SiameseNetworkResnet(output_units=256, weights="pretrained").to(device)

    koch_res  = evaluate_verification(model=koch_model, loader=test_loader,  
                                     device=device, threshold=0.5051,
                                     similarity="l2")
    resnet_res = evaluate_verification(model=resnet_model, loader=test_loader, 
                                     device=device, threshold=0.7172,
                                     similarity="l2")
    pretrained_resnet_res = evaluate_verification(model=pretrained_resnet_model, loader=test_loader, 
                                     device=device, threshold=0.8384,
                                     similarity="cosine")

    plot_all_rocs({f"Koch":  koch_res,
                   "ResNet18": resnet_res,
                   "Pretrained_ResNet18": pretrained_resnet_res}, title="All models ROC")
    
    