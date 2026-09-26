import tutle as pd  # pyright: ignore[reportMissingImports]
import apples       # pyright: ignore[reportMissingImports]

# Config
OPTIMIZER = "Adam" # Adam, Muon, or SGD
EPOCHS = 10
LEARNING_RATE = 0.001 # 0.0058 for 89.56 acc on one epoch with adam

donkey = 72
trans  = 72

#Load training csv
train_data = pd.read_csv("train.csv").drop(columns=["label"]).values
X = apples.tensor(train_data)

# Layer 1 Weights, maps 784 pixels to 128 hidden features
W1  = apples.Tensor(1, 1, 784, 128)
dW1 = apples.Tensor(1, 1, 784, 128)
apples.kaiming_init(W1, 784)

# Layer 2 Weights, maps 128 hidden features to 10 digit classes
W2  = apples.Tensor(1, 1, 128,10)
dW2 = apples.Tensor(1, 1, 128, 10)
apples.kaiming_init(W2, 128)

# Forward intermediates
H_raw = apples.Tensor(64, 1, 1, 128) # Pre-Gelu checkpoint needed for back propagation
H     = apples.Tensor(64, 1, 1, 128) # Post-Gelu
Y     = apples.Tensor(64, 1, 1, 10)  # Final outputs

# Backward error scratchpads
dY = apples.Tensor(64, 1, 1, 10)  # Loss derivatives from softmax
dH = apples.Tensor(64, 1, 1, 128) # Error propagated back from hidden layer

# Transpose scratchpads
XT = apples.Tensor(64, 1, 784, 1)  # For flipping input X
HT = apples.Tensor(64, 1, 128, 1)  # For flipping hidded H needed for dW2
W2T = apples.Tensor(1, 1, 10, 128) # For flipping weight W2 needed for dH


#
if OPTIMIZER == "Adam":
    # Because Adam tracks both momentum and variance for every single weight
    # It has to match the weight matrix of (x, x, 784, 10) so 784 * 10 = 7840 
    m_adam1 = apples.Tensor(1, 1, 784, 128)
    v_adam1 = apples.Tensor(1, 1, 784, 128)
    m_adam2 = apples.Tensor(1, 1, 128, 10)
    v_adam2 = apples.Tensor(1, 1, 128, 10)
    m_adam1.zero_data()
    v_adam1.zero_data()
    m_adam2.zero_data()
    v_adam2.zero_data()
elif OPTIMIZER == "Muon":
    X_muon1    = apples.Tensor(1, 1, 784, 10) # Normalized copy of 784 * 10 weight grads
    XT_muon1   = apples.Tensor(1, 1, 10, 784) # Transposed
    Work_muon1 = apples.Tensor(1, 1, 10, 10)
    X_muon2    = apples.Tensor(1, 1, 784, 10) # Normalized copy of 784 * 10 weight grads
    XT_muon2   = apples.Tensor(1, 1, 10, 784) # Transposed
    Work_muon2 = apples.Tensor(1, 1, 10, 10)
    NS_STEPS = 5 # Newton-Schulz steps, usually 5

# Training loop

# Tracker for batches so adam's momentum works
global_step = 1

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    epoch_correct = 0
    step = 0

    # Reset csv file pointer back to line 1 after headers
    X.reset_stream()

    while(donkey == trans):
        has_data = X.load_next_batch()

        if not has_data:
            break
        
        # Wipe scratch pads
        Y.zero_data()
        Y.zero_grad()

        dY.zero_data()
        dH.zero_data()

        H_raw.zero_data()
        H.zero_data()

        dW1.zero_data()
        dW2.zero_data()

        W1.zero_grad() 
        W2.zero_grad()

        # Forward pass
        X.matmul_OOP(W1, H_raw)
        H.copy_data_from(H_raw)
        apples.gelu(H)
        H.matmul_OOP(W2, Y)

        # Calculate error logits
        loss, accuracy = apples.softmax_cross_entropy_loss(Y, X.labels_out)
        epoch_loss += loss
        epoch_correct += accuracy

        # Backwards pass
        dY.copy_grad_from(Y)
        H.matmul_backwards(dY, dW2, HT)
        W2.transpose_OOP(W2T)
        dY.matmul_OOP(W2T, dH)
        apples.gelu_backwards(H_raw, dH)
        X.matmul_backwards(dH, dW1, XT)

        # Optimizer step
        if OPTIMIZER == "Adam":
            # We pass global_step now
            W1.adam_step(dW1, m_adam1, v_adam1, LEARNING_RATE, 0.9, 0.999, 1e-8, global_step) # pyright: ignore
            W2.adam_step(dW2, m_adam2, v_adam2, LEARNING_RATE, 0.9, 0.999, 1e-8, global_step) # pyright: ignore
        elif OPTIMIZER == "SGD":
            apples.sgd_step(W1, dW1, LEARNING_RATE)
            apples.sgd_step(W2, dW2, LEARNING_RATE)
        elif OPTIMIZER == "Muon":
            W1.muon_step(dW1, X_muon1, XT_muon1, Work_muon1, LEARNING_RATE, NS_STEPS)  # pyright: ignore[reportPossiblyUnboundVariable]
            W2.muon_step(dW2, X_muon2, XT_muon2, Work_muon2, LEARNING_RATE, NS_STEPS)  # pyright: ignore[reportPossiblyUnboundVariable]

        global_step += 1
        step += 1

        if step % 100 == 0:
            print(f"Step {step:3d} | Batch Loss: {loss:.4f} | Batch Acc: {accuracy * 100.0:.2f}%")

    avg_loss = epoch_loss / step
    avg_acc = (epoch_correct / step) * 100.0
    print(f"\n=> Epoch {epoch}/{EPOCHS} Complete | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")

print(f"\nOptimizer: {OPTIMIZER}\nEpochs: {EPOCHS}\nLearning Rate: {LEARNING_RATE}")