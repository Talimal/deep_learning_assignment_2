from models import *
from dataset import *
from evaluation import *
from plotting import *
from experiment_2 import run_with_hyperparam_search
import torch
from constants import HYPERPARAM_GRID
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def find_optimal_batch_size(model, model_name, device, similarity="cosine"):
    best_val = 0
    best_batch_size = 0
    for batch_size in HYPERPARAM_GRID["batch_size"]:
        train_loader, val_loader, test_loader = build_loaders(
            batch_size=batch_size,
        )
        val_res, test_res, best_threshold = evaluate_and_plot(model, model_name, val_loader, 
                                                            test_loader, device=device,
                                                            similarity=similarity)
        print(f"val acc: {val_res['accuracy']:.4f} | "
            f"test acc: {test_res['accuracy']:.4f} | "
            f"threshold: {best_threshold:.4f}")
        if val_res['accuracy'] > best_val:
            best_val = val_res['accuracy']
            best_batch_size = batch_size
    return best_val, best_batch_size
        
    
def run_experiment(model, model_name, device, batch_size, similarity="cosine"):
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=batch_size,
    )
    val_res, test_res, best_threshold = evaluate_and_plot(model, model_name, val_loader, 
                                                          test_loader, device=device,
                                                          similarity=similarity)
    print(f"val acc: {val_res['accuracy']:.4f} | "
          f"test acc: {test_res['accuracy']:.4f} | "
          f"threshold: {best_threshold:.4f}")

    plot_all_rocs({
        "PretrainedResNet18_test":   test_res,
    }, title="PretrainedResNet18_test ROC")
    report_model_stats(model=model, model_name=model_name, device=device, trainable_only=False)
    return model, val_res, test_res


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = ContrastiveLoss()


    resnet_model = SiameseNetworkResnet(output_units=256, weights="pretrained").to(device)
    best_val, best_batch_size = find_optimal_batch_size(model=resnet_model, model_name="pretrained_ResNet18",
                            device=device, similarity="cosine")
    fixed_batch_size = 8
    resnet_model, resnet_val, resnet_test = run_experiment(
        model=resnet_model,
        model_name="pretrained_ResNet18",
        batch_size=fixed_batch_size,
        device=device
    )
    