from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Configuration and reproducibility
SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
BETA = 1.0                      # KL wieight
DEVICE = torch.device("cpu")    # facilitate reproductibility

ACTIVITY_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = Path(__file__).resolve().parents[4]
FIGURE_DIR = ACTIVITY_DIR / "figures"
DATA_DIR = REPOSITORY_DIR / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "font.size": 9,
    }
)


class VAE(nn.Module):
    """Fully connected VAE with a configurable latent dimension"""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        # Defining the encoder (1 x 28 x 28 = 784 - image size)
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # Parameters for image enconding
        self.mu_head = nn.Linear(256, latent_dim)
        self.logvar_head = nn.Linear(256, latent_dim)

        # Defining the decoder (opposite operation)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),   # Pixel value between 0 and 1
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    # Separate randomness from epsilon: allow z gradient calculation
    @staticmethod
    def reparameterize(mu, logvar):
        """Sample z = mu + sigma * epsilon while retaining gradients"""

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        # returns the reconstruction and the parameters used
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, x, mu, logvar, beta=BETA):
    """
    Return summed BCE and closed-form KL, plus batch-mean loss

    BCE is summed over all 784 pixels and KL over every latent dimension
    Dividing their sum by the batch size gives nats per image
    """

    # BCE compares the reconstructed pixels
    reconstruction_sum = F.binary_cross_entropy(reconstruction, x, reduction="sum")

    # Forces the distributions produced to remain close to the standard normal distribution
    kl_sum = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar)
    # Allow the generation fo new images based on sampling

    # Calulate loss based on nats produced by image: relevant to backpropagation
    loss = (reconstruction_sum + beta * kl_sum) / x.shape[0]

    return loss, reconstruction_sum, kl_sum


def load_data():
    transform = transforms.ToTensor()
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train, validation = random_split(train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(SEED),)
    # Training:   50 000 images 
    # Validation: 10 000 images
    # Testing:    10 000 image

    return train, validation, test

# Recreate the loaders for each model: guarantee batches order
def make_loaders(train, validation, test):
    """Create fresh deterministic loaders for one model"""

    train_loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    return train_loader, validation_loader, test_loader


def run_epoch(model, loader, optimizer=None):
    # Optimizer presence defines the training mode
    training = optimizer is not None
    model.train(training)
    # If training: 
    ## gradients are erased and recalculated
    ## parameters are updated

    reconstruction_total = 0.0
    kl_total = 0.0
    sample_count = 0

    # Avoid storing gradient graph
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, _ in loader:
            x = images.to(DEVICE).flatten(start_dim=1)
            if training:
                optimizer.zero_grad()

            reconstruction, mu, logvar = model(x)
            loss, reconstruction_sum, kl_sum = vae_loss(reconstruction, x, mu, logvar)

            if training:
                loss.backward()
                optimizer.step()

            # metrics are aggregated with sums
            reconstruction_total += reconstruction_sum.item()
            kl_total += kl_sum.item()
            sample_count += x.shape[0]

    # calculate simple mean between batches
    return {
        "reconstruction": reconstruction_total / sample_count,
        "kl": kl_total / sample_count,
        "negative_elbo": (reconstruction_total + BETA * kl_total) / sample_count,
    }


def train_model(latent_dim, train, validation, test):
    """Train one VAE without consulting the test set until training ends"""

    # Create the model
    torch.manual_seed(SEED)
    model = VAE(latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, validation_loader, test_loader = make_loaders(
        train, validation, test
    )

    # Store terms saparatedly 
    history = {
        "train_reconstruction": [],
        "train_kl": [],
        "validation_reconstruction": [],
        "validation_kl": [],
    }

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        validation_metrics = run_epoch(model, validation_loader)
        history["train_reconstruction"].append(train_metrics["reconstruction"])
        history["train_kl"].append(train_metrics["kl"])
        history["validation_reconstruction"].append(
            validation_metrics["reconstruction"]
        )
        history["validation_kl"].append(validation_metrics["kl"])
        print(
            f"d={latent_dim:2d} epoch {epoch:02d}/{EPOCHS}: "
            f"train BCE={train_metrics['reconstruction']:.3f}, "
            f"KL={train_metrics['kl']:.3f}; "
            f"val BCE={validation_metrics['reconstruction']:.3f}, "
            f"KL={validation_metrics['kl']:.3f}"
        )

    # The official test set is first evaluated only after training is complete
    test_metrics = run_epoch(model, test_loader)
    return model, history, test_metrics

# BCE and KL (training and validating)
def figure_1(history):
    epochs = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    axes[0].plot(
        epochs, history["train_reconstruction"], label="Train", color="#3B82C4"
    )
    axes[0].plot(
        epochs,
        history["validation_reconstruction"],
        label="Validation",
        color="#B3261E",
    )
    axes[0].set_title("Reconstruction term")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE (nats per image)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, history["train_kl"], label="Train", color="#3B82C4")
    axes[1].plot(
        epochs,
        history["validation_kl"],
        label="Validation",
        color="#B3261E",
    )
    axes[1].set_title("KL term")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("KL divergence (nats per image)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.suptitle("Figure 1: VAE training and validation losses ($d=16$)")
    fig.savefig(FIGURE_DIR / "figure1.png", bbox_inches="tight")
    plt.close(fig)


def first_index_by_digit(dataset):
    indices = {}
    for index in range(len(dataset)):
        _, label = dataset[index]
        if label not in indices:
            indices[label] = index
        if len(indices) == 10:
            break
    return [indices[digit] for digit in range(10)]

# Reconstruct first image for each digit
def figure_2(model, test):
    indices = first_index_by_digit(test)
    images = torch.stack([test[index][0] for index in indices]).to(DEVICE)
    x = images.flatten(start_dim=1)
    model.eval()

    with torch.no_grad():
        mu, _ = model.encode(x)
        reconstructions = model.decode(mu).reshape(-1, 28, 28).cpu()
        # Using mu reduces noise and show the deterministic central reconstruction

    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8), constrained_layout=True)
    for digit in range(10):
        axes[0, digit].imshow(images[digit, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, digit].imshow(
            reconstructions[digit], cmap="gray", vmin=0, vmax=1
        )
        axes[0, digit].set_title(str(digit))
        axes[0, digit].set_xticks([])
        axes[0, digit].set_yticks([])
        axes[1, digit].set_xticks([])
        axes[1, digit].set_yticks([])
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    fig.suptitle("Figure 2: First test image of each digit and reconstruction")
    fig.savefig(FIGURE_DIR / "figure2.png", bbox_inches="tight")
    plt.close(fig)
    return indices


def figure_3(model):
    torch.manual_seed(0)
    z = torch.randn(64, model.latent_dim, device=DEVICE)
    model.eval()
    with torch.no_grad():
        samples = model.decode(z).reshape(-1, 28, 28).cpu()

    fig, axes = plt.subplots(8, 8, figsize=(8, 8), constrained_layout=True)
    for axis, sample in zip(axes.flat, samples):
        axis.imshow(sample, cmap="gray", vmin=0, vmax=1)
        axis.set_xticks([])
        axis.set_yticks([])
    
    fig.suptitle("Figure 3: Samples decoded from $z \\sim \\mathcal{N}(0,I)$")
    fig.supxlabel("Prior sample columns")
    fig.supylabel("Prior sample rows")
    fig.savefig(FIGURE_DIR / "figure3.png", bbox_inches="tight")

    plt.close(fig)


def encode_dataset(model, loader):
    model.eval()
    mus, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            x = images.to(DEVICE).flatten(start_dim=1)
            mu, _ = model.encode(x)
            mus.append(mu.cpu())
            labels.append(batch_labels)
    return torch.cat(mus).numpy(), torch.cat(labels).numpy()

# For each test image, mu is captured
def figure_4(model, test_loader):
    means, labels = encode_dataset(model, test_loader)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            means[mask, 0],
            means[mask, 1],
            s=8,
            alpha=0.45,
            color=colors(digit),
            label=str(digit),
            edgecolors="none",
        )
    ax.set_title("Figure 4: Encoded MNIST test set ($d=2$)")
    ax.set_xlabel("Latent mean $\\mu_1$")
    ax.set_ylabel("Latent mean $\\mu_2$")
    ax.legend(title="Digit", ncols=2, markerscale=2.2)
    ax.grid(alpha=0.15)
    fig.savefig(FIGURE_DIR / "figure4.png", bbox_inches="tight")
    plt.close(fig)
    return means, labels

# Grid using quantis 
def figure_5(model):
    quantiles = np.linspace(0.05, 0.95, 15)
    coordinates = norm.ppf(quantiles) # Transform probabilities in coordinates
    xx, yy = np.meshgrid(coordinates, coordinates[::-1])

    z = torch.tensor(
        np.column_stack([xx.ravel(), yy.ravel()]),
        dtype=torch.float32,
        device=DEVICE,
    )
    model.eval()
    with torch.no_grad():
        decoded = model.decode(z).reshape(15, 15, 28, 28).cpu().numpy()

    canvas = np.zeros((15 * 28, 15 * 28))
    for row in range(15):
        for column in range(15):
            canvas[
                row * 28 : (row + 1) * 28,
                column * 28 : (column + 1) * 28,
            ] = decoded[row, column]

    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    extent = [coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]]
    ax.imshow(canvas, cmap="gray", vmin=0, vmax=1, extent=extent, origin="upper")
    ax.set_title("Figure 5: Decoded latent plane on an equal-prior-mass grid")
    ax.set_xlabel("Latent coordinate $z_1$")
    ax.set_ylabel("Latent coordinate $z_2$")
    ticks = coordinates[[0, 3, 7, 11, 14]]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.savefig(FIGURE_DIR / "figure5.png", bbox_inches="tight")
    plt.close(fig)

# Compare interpolations
def figure_6(model, test, index_a=2, index_b=0):
    image_a, label_a = test[index_a]
    image_b, label_b = test[index_b]
    if label_a == label_b:
        raise ValueError("Interpolation endpoints must have different labels")

    x_a = image_a.flatten().to(DEVICE)
    x_b = image_b.flatten().to(DEVICE)
    model.eval()
    with torch.no_grad():
        mu_a, _ = model.encode(x_a.unsqueeze(0))
        mu_b, _ = model.encode(x_b.unsqueeze(0))
        t = torch.linspace(0, 1, 10, device=DEVICE).unsqueeze(1)
        pixel_interpolation = (1 - t) * x_a + t * x_b
        latent_interpolation = (1 - t) * mu_a + t * mu_b
        decoded_interpolation = model.decode(latent_interpolation)

    rows = [
        pixel_interpolation.reshape(-1, 28, 28).cpu(),
        decoded_interpolation.reshape(-1, 28, 28).cpu(),
    ]
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8), constrained_layout=True)
    for row_index, images in enumerate(rows):
        for column, image in enumerate(images):
            axes[row_index, column].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            axes[row_index, column].set_title(f"{column / 9:.2f}")
    axes[0, 0].set_ylabel("Pixel space")
    axes[1, 0].set_ylabel("Latent space")
    fig.suptitle(
        "Figure 6: Interpolation from "
        f"test[{index_a}] = {label_a} to test[{index_b}] = {label_b}"
    )
    fig.supxlabel("Interpolation coefficient $t$")
    fig.savefig(FIGURE_DIR / "figure6.png", bbox_inches="tight")
    plt.close(fig)
    return (index_a, int(label_a)), (index_b, int(label_b))


def print_metrics(name, metrics):
    print(
        f"{name}: reconstruction={metrics['reconstruction']:.4f}, "
        f"KL={metrics['kl']:.4f}, "
        f"negative ELBO={metrics['negative_elbo']:.4f} nats/image"
    )


def main():
    torch.use_deterministic_algorithms(True)
    train, validation, test = load_data()

    print("=== EXERCISE 1: d=16 ===")
    model_16, history_16, test_16 = train_model(
        16, train, validation, test
    )
    print_metrics("d=16 test", test_16)
    figure_1(history_16)
    reconstruction_indices = figure_2(model_16, test)
    figure_3(model_16)

    print("\n=== EXERCISE 2: d=2 ===")
    model_2, history_2, test_2 = train_model(2, train, validation, test)
    print_metrics("d=2 test", test_2)
    _, _, test_loader = make_loaders(train, validation, test)
    figure_4(model_2, test_loader)
    figure_5(model_2)
    endpoints = figure_6(model_16, test)

    print(f"First test indices for digits 0-9: {reconstruction_indices}")
    print(f"Interpolation endpoints: {endpoints}")
    print(
        "Validation KL d=16: "
        f"epoch 1={history_16['validation_kl'][0]:.4f}, "
        f"epoch {EPOCHS}={history_16['validation_kl'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
