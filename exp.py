import os
import time
import copy
import torch
# Enable MPS fallback for unhandled ops & enable high-performance memory allocator
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# torch.backends.mps.enable_performance_mode() # PyTorch 2.1+ optimization
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# -------------------------------------------------------------------
# Device & Configuration
# -------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print(" Running on Apple Silicon GPU via MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(" Running on NVIDIA CUDA GPU")
        return torch.device("cuda")
    else:
        print(" Running on CPU")
        return torch.device("cpu")

DEVICE = get_device()
NUM_WORKERS = 0
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.1
NOISE_RATE = 0.4
SEEDS = [42, 123, 999]

def get_resnet18():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)

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
# Single Seed Execution with Checkpoint Saving
# -------------------------------------------------------------------
def run_single_seed(seed):
    print(f"\n=======================================================")
    print(f" Starting Proposed Method Run | Seed: {seed}")
    print(f"=======================================================")
    
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    num_train = len(full_train)
    noisy_labels = np.array(full_train.targets)
    mask = np.random.rand(num_train) < NOISE_RATE
    for i in range(num_train):
        if mask[i]:
            noisy_labels[i] = np.random.choice([l for l in range(10) if l != full_train.targets[i]])
    full_train.targets = noisy_labels.tolist()

    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:40000], indices[40000:]

    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = get_resnet18()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_model_weights = None
    best_epoch = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        if (epoch + 1) % 50 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | Val Acc: {val_acc:.2f}% | Best Val: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Save Checkpoint to Disk
    os.makedirs("./checkpoints", exist_ok=True)
    checkpoint_path = f"./checkpoints/proposed_peak_seed_{seed}.pth"
    torch.save(best_model_weights, checkpoint_path)
    print(f"\n Saved peak model weights to: {checkpoint_path}")

    # Post-Hoc Unlearning
    model.load_state_dict(best_model_weights)
    peak_test_acc = evaluate(model, test_loader)

    unlearn_optimizer = optim.SGD(model.parameters(), lr=0.001)
    model.train()
    for _ in range(5):
        for (v_in, v_tar), (t_in, t_tar) in zip(val_loader, train_loader):
            v_in, v_tar = v_in.to(DEVICE), v_tar.to(DEVICE)
            t_in, t_tar = t_in.to(DEVICE), t_tar.to(DEVICE)

            unlearn_optimizer.zero_grad()
            loss = criterion(model(v_in), v_tar) - 0.05 * criterion(model(t_in), t_tar)
            loss.backward()
            unlearn_optimizer.step()

    final_test_acc = evaluate(model, test_loader)
    elapsed_time = (time.time() - start_time) / 60

    print(f"Seed {seed} Complete | Runtime: {elapsed_time:.2f} mins | Peak Acc: {peak_test_acc:.2f}% | Unlearned Acc: {final_test_acc:.2f}%")
    return peak_test_acc, final_test_acc

if __name__ == "__main__":
    peak_results, unlearned_results = [], []
    for seed in SEEDS:
        peak_acc, unlearned_acc = run_single_seed(seed)
        peak_results.append(peak_acc)
        unlearned_results.append(unlearned_acc)

    print("\n" + "="*60)
    print("      MULTI-SEED BENCHMARK STATISTICAL SUMMARY (N=3)       ")
    print("="*60)
    print(f"Peak Checkpoint Accuracy:      {np.mean(peak_results):.2f}% ± {np.std(peak_results):.2f}%")
    print(f"Final Unlearned Test Accuracy: {np.mean(unlearned_results):.2f}% ± {np.std(unlearned_results):.2f}%")
    print("="*60)