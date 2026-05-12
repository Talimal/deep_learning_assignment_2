from evaluation import *
from dataset import *
from models import *
import constants


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=constants.BATCH_SIZE,
    )
    episodes_2  = generate_oneshot_episodes(test_loader.dataset, N=2,  num_episodes=400)
    episodes_5  = generate_oneshot_episodes(test_loader.dataset, N=5,  num_episodes=400)
    episodes_20 = generate_oneshot_episodes(test_loader.dataset, N=20, num_episodes=400)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    koch_model = SiameseNetwork()
    koch_weights_path = "Koch/weights_SEED=42.pth"
    koch_model.load_state_dict(torch.load(koch_weights_path))
    koch_model = koch_model.to(device)

    resnet_model = SiameseNetworkResnet()
    resnet_weights_path = "ResNet18/weights_SEED=42.pth"
    resnet_model.load_state_dict(torch.load(resnet_weights_path))
    resnet_model = resnet_model.to(device)

    pretrained_resnet_model = SiameseNetworkResnet(output_units=256, weights="pretrained").to(device)

    oneshot_res = {}
    for model, name, similarity in [(koch_model, "Koch", "l2"),
                                    (resnet_model, "ResNet18", "l2"),
                                    (pretrained_resnet_model, "PretrainedResNet18", "cosine")]:
        oneshot_res[name] = {}
        for N, episodes in [(2, episodes_2), (5, episodes_5), (20, episodes_20)]:
            acc = evaluate_oneshot(model, test_loader.dataset, episodes, N, device, similarity)
            oneshot_res[name][N] = acc
            print(f"{name} | {N}-way one-shot acc: {acc:.4f}")
    plot_oneshot_results(results_dict=oneshot_res)