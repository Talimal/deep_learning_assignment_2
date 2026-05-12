from evaluation import *
from dataset import *
from models import *
from plotting import *
import constants
import random

seed = constants.SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    fixed_batch_size = 8
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=fixed_batch_size,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    pretrained_resnet_model = SiameseNetworkResnet(output_units=256, weights="pretrained").to(device)

    plot_embedding_projection(model=pretrained_resnet_model, dataset=test_loader.dataset,
                              model_name="Pretrained_ResNet18", method="umap",
                              n_identities=25,
                              samples_per_identity=5,
                              device=device)