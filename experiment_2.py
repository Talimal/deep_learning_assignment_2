from models import *
from dataset import *
from evaluation import *
from plotting import *
import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def train_model(model, model_name, criterion, lr, best_threshold):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, wall_time = train(
        model=model,
        train_epochs=constants.TRAIN_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        best_threshold=best_threshold,
    )
    return history, wall_time


def run_experiment(model_class, model_name, lr, best_threshold,
                   margin, batch_size, device, similarity="l2"):
    criterion = ContrastiveLoss(margin=margin)
    model = model_class(output_units=256).to(device)

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=batch_size,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, wall_time = train(
        model=model,
        train_epochs=constants.TRAIN_EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        best_threshold=best_threshold,
        train_loader=train_loader,
        val_loader=val_loader
    )

    report_model_stats(model, model_name, wall_time, device)
    val_res, test_res, best_threshold = evaluate_and_plot(
        model, model_name, val_loader=val_loader, test_loader=test_loader,
        similarity=similarity, device=device
    )

    print(f"\n{model_name} | time: {wall_time:.2f}s | "
          f"val acc: {val_res['accuracy']:.4f} | "
          f"test acc: {test_res['accuracy']:.4f} | "
          f"threshold: {best_threshold:.4f}")

    return model, history, val_res, test_res



def run_with_hyperparam_search(model_class, model_name, device, similarity="l2"):
    best_params, best_threshold = hyperparam_search(
        model_class=model_class,
        model_name=model_name,
        model_kwargs={"output_units": 256},
        device=device,
        train_epochs=constants.TRAIN_EPOCHS,
        similarity=similarity
    )
    return run_experiment(model_class, model_name,
                          lr=best_params["lr"],
                          best_threshold=best_threshold,
                          margin=best_params["margin"],
                          batch_size=best_params["batch_size"],
                          device=device)



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iteration 1 for hyperparameters search:

    koch_model, koch_history, koch_val, koch_test = run_with_hyperparam_search(SiameseNetwork,
                                                                               "Koch",
                                                                               device=device)
    resnet_model, resnet_history, resnet_val, resnet_test = run_with_hyperparam_search(SiameseNetworkResnet, 
                                                                                       "ResNet18",
                                                                                       device=device)

    # Iteration 2 for running with best params and getting plots:

    koch_model, koch_history, koch_val, koch_test = run_experiment(SiameseNetwork, "Koch",
                          lr=0.0001,
                          best_threshold=0.6162,
                          margin=2.0,
                          batch_size=16,
                          device=device)
    resnet_model, resnet_history, resnet_val, resnet_test = run_experiment(SiameseNetworkResnet, "ResNet18",
                          lr=0.01,
                          best_threshold=0.7172,
                          margin=1.0,
                          batch_size=4, 
                          device=device)
    plot_history(koch_history, model_name="Koch")      
    plot_history(resnet_history, model_name="ResNet18")                                                                          
    plot_all_rocs({
        "Koch_test":   koch_test,
        "ResNet18_test": resnet_test,
    }, title="Koch VS ResNet18 ROC")

    # Iteration 3: pick the best model and run with another 2 seeds (change in constants)
    # resnet_model, resnet_history, resnet_val, resnet_test = run_experiment(SiameseNetworkResnet, 
    #                                                                        "ResNet18",
    #                       lr=0.01,
    #                       best_threshold=0.7172,
    #                       margin=1.0,
    #                       batch_size=4, 
    #                       device=device)
    # plot_history(resnet_history, model_name="Koch")      
    # plot_all_rocs({
    #     "ResNet18":   resnet_test,
    # }, title="ResNet18 ROC")

