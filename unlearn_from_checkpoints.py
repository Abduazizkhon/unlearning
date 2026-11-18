import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# -------------------------------------------------------------------
# Device & Configuration
# -------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"Using compute device: {DEVICE}")

BATCH_SIZE = 128
NOISE_RATE = 0.4        # 40% Injected Symmetric Noise
UNLEARN_EPOCHS = 10     # Number of unlearning epochs
GAMMA = 0.01            # Gradient ascent underweight factor
UNLEARN_LR = 0.001      # Fine-tuning learning rate for unlearning
NUM_WORKERS = 0
SEEDS = [42, 123, 999]  # Your 3 seed checkpoints

# -------------------------------------------------------------------
# Architecture Definition (ResNet-18)
# -------------------------------------------------------------------
def get_resnet18():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

# -------------------------------------------------------------------
# Pure Evaluation Function (No Gradients)
# -------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total

# -------------------------------------------------------------------
# Unlearning Single Seed Function
# -------------------------------------------------------------------
def unlearn_single_seed(seed):
    print(f"\n=======================================================")
    print(f" Starting Unlearning Evaluation | Seed: {seed}")
    print(f"=======================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Reconstruct Dataset & Noise Injection with exact seed reproducibility
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_train = len(full_train)
    noisy_labels = np.array(full_train.targets)
    mask = np.random.rand(num_train) < NOISE_RATE
    for i in range(num_train):
        if mask[i]:
            noisy_labels[i] = np.random.choice([l for l in range(10) if l != full_train.targets[i]])
    full_train.targets = noisy_labels.tolist()

    # Reconstruct 40,000 Train / 10,000 Validation Split
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:40000], indices[40000:]

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 1. Load Pre-Saved Peak Weights from Checkpoints Directory
    checkpoint_path = f"./checkpoints/proposed_peak_seed_{seed}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = get_resnet18()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    print(f" Loaded clean peak checkpoint from: {checkpoint_path}")

    # Evaluate Peak Checkpoint Accuracy on Clean Test Set
    peak_test_acc = evaluate(model, test_loader)
    peak_val_acc = evaluate(model, val_loader)
    print(f" Peak Checkpoint -> Val Acc: {peak_val_acc:.2f}% | Test Acc: {peak_test_acc:.2f}%")

    # 2. Setup Selective Training Set Unlearning (Zero Validation Backprop)
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=UNLEARN_LR, momentum=0.9, weight_decay=5e-4)

    start_time = time.time()
    best_val_acc = peak_val_acc
    best_unlearned_weights = copy.deepcopy(model.state_dict())

    model.train()
    for epoch in range(1, UNLEARN_EPOCHS + 1):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            per_sample_losses = criterion_none(outputs, targets)

            # Identify small-loss (clean) vs high-loss (noisy) samples INSIDE training batch
            threshold = torch.quantile(per_sample_losses.detach(), 1.0 - NOISE_RATE)
            clean_mask = (per_sample_losses <= threshold).float()
            noisy_mask = (per_sample_losses > threshold).float()

            # GD on clean train samples + Underweighted Gradient Ascent on noisy train samples
            loss = (clean_mask * per_sample_losses - GAMMA * noisy_mask * per_sample_losses).mean()

            loss.backward()
            optimizer.step()

        # Monitor Validation Set WITHOUT computing gradients
        val_acc = evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_unlearned_weights = copy.deepcopy(model.state_dict())

        print(f" Unlearn Epoch [{epoch:02d}/{UNLEARN_EPOCHS}] | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    # 3. Final Evaluation on Clean Test Set
    model.load_state_dict(best_unlearned_weights)
    final_test_acc = evaluate(model, test_loader)
    elapsed_time = (time.time() - start_time) / 60.0

    print(f" Seed {seed} Complete | Time: {elapsed_time:.2f} mins")
    print(f" Peak Test Acc: {peak_test_acc:.2f}%  --->  Unlearned Test Acc: {final_test_acc:.2f}%")

    return peak_test_acc, final_test_acc

# -------------------------------------------------------------------
# Multi-Seed Execution & Statistical Summary
# -------------------------------------------------------------------
if __name__ == "__main__":
    peak_results, unlearned_results = [], []

    for seed in SEEDS:
        peak_acc, unlearned_acc = unlearn_single_seed(seed)
        peak_results.append(peak_acc)
        unlearned_results.append(unlearned_acc)

    print("\n" + "="*65)
    print("   LEAKAGE-FREE MULTI-SEED UNLEARNING BENCHMARK SUMMARY (N=3)   ")
    print("="*65)
    print(f" Peak Checkpoint Accuracy : {np.mean(peak_results):.2f}% ± {np.std(peak_results):.2f}% | Seeds: {[round(s, 2) for s in peak_results]}")
    print(f" Final Unlearned Accuracy : {np.mean(unlearned_results):.2f}% ± {np.std(unlearned_results):.2f}% | Seeds: {[round(s, 2) for s in unlearned_results]}")
    print(f" Net Accuracy Improvement : +{(np.mean(unlearned_results) - np.mean(peak_results)):.2f}%")
    print("="*65)