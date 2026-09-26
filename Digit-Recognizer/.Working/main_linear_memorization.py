import tutle as pd  # pyright: ignore[reportMissingImports]
import apples       # pyright: ignore[reportMissingImports]

# Config
OPTIMIZER = "Adam" # Adam, Muon, or SGD
EPOCHS = 6
LEARNING_RATE = 1

X = apples.Tensor(64, 1, 28, 28, csv_stream_path="train.csv")

# Allocate Weight Matrix and apply kaiming
W = apples.Tensor(1, 1, 784, 10)
# https://app.notion.com/p/Something-372620b9962b8005bdb1d7eeccfb3b4c
#                 (Tensor, fan_in)
apples.kaiming_init(W, 784)

# Allocate intermedite backpropagation scratchpads
Y  = apples.Tensor(64, 1, 1, 10) # Batch of 64 so 64 images at the same time and 10 classification digits for 0-9
dY = apples.Tensor(64, 1, 1, 10)
dW = apples.Tensor(1, 1, 784, 10)
XT = apples.Tensor(64, 1, 784, 1) # Input matrix X has shape (64, 784) and to preform the mul we has to transpose so it ends up as (784, 64)

#
if OPTIMIZER == "Adam":
    # Because Adam tracks both momentum and variance for every single weight
    # It has to match the weight matrix of (x, x, 784, 10) so 784 * 10 = 7840 
    m_adam = apples.Tensor(1, 1, 784, 10)
    v_adam = apples.Tensor(1, 1, 784, 10)
    m_adam.zero_data()
    v_adam.zero_data()
elif OPTIMIZER == "Muon":
    X_muon    = apples.Tensor(1, 1, 784, 10) # Normalized copy of 784 * 10 weight grads
    XT_muon   = apples.Tensor(1, 1, 10, 784) # Transposed
    # A = M * N
    # B = N * M
    # C = N * N
    Work_muon = apples.Tensor(1, 1, 10, 10)
    NS_STEPS = 5 # Newton-Schulz steps, usually 5

# Training loop
for epoch in range(1, EPOCHS + 1):
    # Wipe scratch pads
    Y.zero_data()
    Y.zero_grad()
    dY.zero_data()
    dW.zero_data()
    W.zero_grad()

    # Forward pass
    X.matmul_OOP(W, Y) # X * W = Y
    loss, accuracy = apples.softmax_cross_entropy_loss(Y, X.labels_out) # Y is the guess the network took and X.labels_out is the correct answers

    # Backwards pass
    dY.copy_grad_from(Y)
    # Shift softmax derivatives into dY.data
    X.matmul_backwards(dY, dW, XT)

    # Optimizer step
    if OPTIMIZER == "Adam":
        # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/#:~:text=Key%20Parameters
        W.adam_step(dW, m_adam, v_adam, LEARNING_RATE, 0.9, 0.999, 1e-8, epoch) # What are these random numbers  # pyright: ignore[reportPossiblyUnboundVariable]
    elif OPTIMIZER == "SGD":
        apples.sgd_step(W, dW, LEARNING_RATE)
    elif OPTIMIZER == "Muon":
        W.muon_step(dW, X_muon, XT_muon, Work_muon, LEARNING_RATE, NS_STEPS)  # pyright: ignore[reportPossiblyUnboundVariable]

    print(f"Epoch {epoch}/{EPOCHS} -> Loss: {loss:.6f} | Accuracy: {accuracy * 100.0:.3f}")

print(f"\nOptimizer: {OPTIMIZER}\nEpochs: {EPOCHS}\nLearning Rate: {LEARNING_RATE}")

print(f"\nInput Matrix Shape: {X.shape}")
print(f"64-Byte Strided Width: {X._c_tensor.stride_w} floats")
print(f"Total Allocated Size: {X._c_tensor.total_size} elements") # 64 * 784 = 50176