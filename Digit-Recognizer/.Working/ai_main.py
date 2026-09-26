import pandas as pd
import torch
import torch.nn.functional as F
import time
import math

# ---------------------------------------------------------
# VIBECODER CONFIGURATION
# ---------------------------------------------------------
EPOCHS = 60
BATCH_SIZE = 64
MAX_LR = 0.05       # AdamW is much more sensitive than SGD
WEIGHT_DECAY = 0.01

print("\n[1/3] Loading Data & Forcing Hardware Constraints...")
# 1. Load data directly into a pinned PyTorch tensor
# We use float32 for data loading, then cast later for speed
df = pd.read_csv("/home/apologizes/kaggle/Digit-Recognizer_old/train.csv")
y_train = torch.tensor(df["label"].values, dtype=torch.long)
X_train = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32) / 255.0

# Pre-calculate steps for the scheduler
num_batches = math.ceil(len(X_train / BATCH_SIZE))
total_steps = EPOCHS * num_batches

# 2. Define a hyper-minimal, compiled model
class FastMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(784, 10, bias=False)
        # Kaiming initialization
        torch.nn.init.kaiming_normal_(self.layer.weight, nonlinearity='linear')

    def forward(self, x):
        return self.layer(x)

model = FastMLP()

# --- THE SECRET WEAPONS ---
# 1. Fuse operations into C++ kernels automatically
compiled_model = torch.compile(model) 
# 2. Use AdamW instead of raw SGD
optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
# 3. OneCycleLR ramps the learning rate up and down for insane convergence speeds
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, total_steps=total_steps, pct_start=0.3
)

print(f"\n[2/3] Launching {EPOCHS}-Epoch Hyper-Optimized Loop...")
start_time = time.time()

# ---------------------------------------------------------
# THE TRAINING LOOP
# ---------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    # Process in batches
    for i in range(0, len(X_train), BATCH_SIZE):
        X_batch = X_train[i:i+BATCH_SIZE]
        y_batch = y_train[i:i+BATCH_SIZE]
        
        optimizer.zero_grad(set_to_none=True) # Faster than .zero_grad()
        
        # Use bfloat16 for 2x memory throughput 
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = compiled_model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy on the fly
        epoch_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    # Print epoch stats
    avg_loss = epoch_loss / num_batches
    accuracy = (correct / total) * 100.0
    print(f"  Epoch {epoch}/{EPOCHS} -> Loss: {avg_loss:.6f} | Accuracy: {accuracy:.3f}%")

end_time = time.time()

print("\n[3/3] Diagnostics & Benchmarks")
print(f"Optimizer: AdamW (OneCycleLR)")
print(f"Max Learning Rate: {MAX_LR}")
print(f"Precision: bfloat16")
print(f"\nTotal Execution Time: {end_time - start_time:.4f} seconds!")