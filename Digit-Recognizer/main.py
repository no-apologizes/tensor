import apples                 # pyright: ignore[reportMissingImports]
from magicalponies import ffi # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as matt

# Config
OPTIMIZER = "Adam" # Adam or SGD
EPOCHS = 10
LEARNING_RATE = 0.01

# Stream data
X = apples.Tensor(64, 1, 28, 28, csv_stream_path="train.csv")

# Layer weight matrices
W_conv = apples.Tensor(16, 1, 3, 3)
dW_conv = apples.Tensor(16, 1, 3, 3)
apples.kaiming_init(W_conv, 1 * 3 * 3)

W_dense  = apples.Tensor(1, 1, 2704, 10)
dW_dense = apples.Tensor(1, 1, 2704, 10)
apples.kaiming_init(W_dense, 2704)

# Forward intermediate checkpoints
Conv_Raw = apples.Tensor(64, 16, 26, 26)
Conv_Act = apples.Tensor(64, 16, 26, 26)
Pool_Out = apples.Tensor(64, 16, 13, 13)
Flat_Out = apples.Tensor(64, 1, 1, 2704)
Y        = apples.Tensor(64, 1, 1, 10)

# Backward grad scratchpads
dY        = apples.Tensor(64, 1, 1, 10)
dFlat_Out = apples.Tensor(64, 1, 1, 2704)
dPool_Out = apples.Tensor(64, 16, 13, 13)
dConv_Out = apples.Tensor(64, 16, 26, 26)

# Math transpose scratchpads
Flat_Out_T = apples.Tensor(64, 1, 2704, 1)
W_dense_T = apples.Tensor(1, 1, 10, 2704)

# MaxPool map tracking
pool_indices = ffi.new("size_t[]", Pool_Out._c_tensor.total_size)

# Optimizer parameter setup
if OPTIMIZER == "Adam" or OPTIMIZER == "SGD":
    m_adam_conv  = apples.Tensor(16, 1, 3, 3)
    v_adam_conv  = apples.Tensor(16, 1, 3, 3)
    
    m_adam_dense = apples.Tensor(1, 1, 2704, 10)
    v_adam_dense = apples.Tensor(1, 1, 2704, 10)
    
    m_adam_conv.zero_data()
    m_adam_dense.zero_data()
    
    v_adam_conv.zero_data()
    v_adam_dense.zero_data()

# Plot stuff
# 10 epochs * 6 logs per epoch (every 100 steps)
LOGS_PER = 6
SLOTS = EPOCHS * LOGS_PER
plot_batch_steps  = apples.Tensor(1, 1, 1, SLOTS)
plot_batch_losses = apples.Tensor(1, 1, 1, SLOTS)
plot_batch_accs   = apples.Tensor(1, 1, 1, SLOTS)

# 10 epochs = 10 static data slots
plot_epoch_steps  = apples.Tensor(1, 1, 1, EPOCHS)
plot_train_losses = apples.Tensor(1, 1, 1, EPOCHS)
plot_train_accs   = apples.Tensor(1, 1, 1, EPOCHS)
plot_val_losses   = apples.Tensor(1, 1, 1, EPOCHS)
plot_val_accs     = apples.Tensor(1, 1, 1, EPOCHS)

batch_log_idx = 0
epoch_log_idx = 0

# Training loop
global_step = 1

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    epoch_correct = 0
    train_steps = 0

    X.reset_stream()

    # Train on the first 600 batches (38,400 images)
    for step in range(1, 601):
        has_data = X.load_next_batch()
        if not has_data:
            break
        
        # --- Hot Path Memory Clear ---
        Y.zero_data()        
        Y.zero_grad()

        dY.zero_data()
        dFlat_Out.zero_data()
        dFlat_Out.zero_grad()
        dPool_Out.zero_grad()
        dConv_Out.zero_grad()
        
        Conv_Raw.zero_data()
        Conv_Act.zero_data()
        Pool_Out.zero_data()
        Flat_Out.zero_data()
        
        dW_conv.zero_data()
        dW_conv.zero_grad()
        dW_dense.zero_data()
        dW_dense.zero_grad()
        
        W_conv.zero_grad()
        W_dense.zero_grad()

        # Forward pass
        X.conv2d(W_conv, Conv_Raw)
        Conv_Act.copy_data_from(Conv_Raw)
        apples.gelu(Conv_Act)
        Conv_Act.maxpool2d(Pool_Out, pool_indices)
        Pool_Out.flatten_copy(Flat_Out)
        Flat_Out.matmul_OOP(W_dense, Y)

        # Loss
        #   |   \  |
        #   |   \  |  |
        # --------------
        #  |  | \ |
        #  |  | \ |____
        loss, accuracy = apples.softmax_cross_entropy_loss(Y, X.labels_out)
        epoch_loss += loss
        epoch_correct += accuracy
        train_steps += 1

        # Backpropagation pass
        dY.copy_grad_from(Y)
        Flat_Out.matmul_backwards(dY, dW_dense, Flat_Out_T)
        W_dense.matmul_grad_input(dY, dFlat_Out, W_dense_T)
        
        ffi.memmove(dFlat_Out._c_tensor.grad, dFlat_Out._c_tensor.data, dFlat_Out._c_tensor.total_size * ffi.sizeof("float"))
        dFlat_Out.unflatten_copy(dPool_Out)
        dPool_Out.maxpool2d_backwards(dConv_Out, pool_indices)
        apples.gelu_backwards(Conv_Raw, dConv_Out)
        X.strided_weight_conv2d_backwards(dConv_Out, dW_conv)
        
        ffi.memmove(dW_conv._c_tensor.data, dW_conv._c_tensor.grad, dW_conv._c_tensor.total_size * ffi.sizeof("float"))

        # Optimizer step
        if OPTIMIZER == "Adam":
            W_conv.adam_step(dW_conv, m_adam_conv, v_adam_conv, LEARNING_RATE, 0.9, 0.999, 1e-8, global_step) # pyright: ignore
            W_dense.adam_step(dW_dense, m_adam_dense, v_adam_dense, LEARNING_RATE, 0.9, 0.999, 1e-8, global_step) # pyright: ignore
        elif OPTIMIZER == "SGD":
            apples.sgd_step(W_conv, dW_conv, LEARNING_RATE)
            apples.sgd_step(W_dense, dW_dense, LEARNING_RATE)

        if step % 100 == 0:
            print(f"Step {step:3d} | Batch Loss: {loss:.4f} | Batch Acc: {accuracy * 100.0:.2f}%")

            plot_batch_steps[batch_log_idx] = global_step   # steps
            plot_batch_losses[batch_log_idx] = loss         # loss
            plot_batch_accs[batch_log_idx] = accuracy * 100 # accurancy
            batch_log_idx += 1
            global_step += 1

    avg_train_loss = epoch_loss / train_steps
    avg_train_acc = (epoch_correct / train_steps) * 100.0

    # Validation pass on the last 56 batches
    val_loss = 0.0
    val_correct = 0
    val_steps = 0

    while(True):
        has_val_data = X.load_next_batch() # Keeps reading from image 38,401 onwards
        if not has_val_data:
            break
        
        # Wipe logit outputs so the GEMM accumulator starts fresh
        Y.zero_data()
        Conv_Raw.zero_data()

        # Forward pass only
        X.conv2d(W_conv, Conv_Raw)
        Conv_Act.copy_data_from(Conv_Raw)
        apples.gelu(Conv_Act)
        Conv_Act.maxpool2d(Pool_Out, pool_indices)
        Pool_Out.flatten_copy(Flat_Out)
        Flat_Out.matmul_OOP(W_dense, Y)

        loss, accuracy = apples.softmax_cross_entropy_loss(Y, X.labels_out)
        val_loss += loss
        val_correct += accuracy
        val_steps += 1

    avg_val_loss = val_loss / val_steps
    avg_val_acc = (val_correct / val_steps) * 100.0

    print("-------------------------------------------------")
    print(f"\n=> Epoch {epoch}/{EPOCHS} Complete")
    print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.3f}%")
    print(f"   Valid Loss: {avg_val_loss:.4f} | Valid Acc: {avg_val_acc:.3f}%\n")
    print("-------------------------------------------------")

    plot_epoch_steps[epoch_log_idx] = global_step - 1
    plot_train_losses[epoch_log_idx] = avg_train_loss
    plot_train_accs[epoch_log_idx] = avg_train_acc
    plot_val_losses[epoch_log_idx] = avg_val_loss
    plot_val_accs[epoch_log_idx] = avg_val_acc
    epoch_log_idx += 1

print(f"\nOptimizer: {OPTIMIZER}\nEpochs: {EPOCHS}\nLearning Rate: {LEARNING_RATE}")

matt.rcParams['text.color'] = '#E0E0E6'
matt.rcParams['axes.labelcolor'] = '#E0E0E6'
matt.rcParams['xtick.color'] = '#A0A0A8'
matt.rcParams['ytick.color'] = '#A0A0A8'

fig, (ax1, ax2) = matt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#111116') # Set external window frame backdrop

ax1.set_facecolor('#181820')
ax1.plot(plot_batch_steps[:batch_log_idx], plot_batch_losses[:batch_log_idx], label="Batch Loss", color="#4D94FF", alpha=0.55, linestyle="--", linewidth=1.0)
ax1.plot(plot_epoch_steps[:epoch_log_idx], plot_train_losses[:epoch_log_idx], label="Train Loss (Epoch)", color="#1A66FF", linewidth=2.5, marker="o", markersize=6)
ax1.plot(plot_epoch_steps[:epoch_log_idx], plot_val_losses[:epoch_log_idx], label="Valid Loss (Epoch)", color="#FF944D", linewidth=2.5, marker="s", markersize=6)
ax1.set_title("Loss Convergence Curve", fontsize=12, fontweight="bold", pad=10, color="#FFFFFF")
ax1.set_xlabel("Global Step (Cumulative)", fontsize=10)
ax1.set_ylabel("Cross-Entropy Loss Metric", fontsize=10)
ax1.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
legend1 = ax1.legend(facecolor='#181820', edgecolor='#2A2A35')
matt.setp(legend1.get_texts(), color='#E0E0E6')

ax2.set_facecolor('#181820')
ax2.plot(plot_batch_steps[:batch_log_idx], plot_batch_accs[:batch_log_idx], label="Batch Accuracy", color="#FF6B8B", alpha=0.55, linestyle="--", linewidth=1.0)
ax2.plot(plot_epoch_steps[:epoch_log_idx], plot_train_accs[:epoch_log_idx], label="Train Accuracy (Epoch)", color="#FF1A4D", linewidth=2.5, marker="o", markersize=6)
ax2.plot(plot_epoch_steps[:epoch_log_idx], plot_val_accs[:epoch_log_idx], label="Valid Accuracy (Epoch)", color="#00CC66", linewidth=2.5, marker="s", markersize=6)
ax2.set_title("Accuracy Trajectory Profile", fontsize=12, fontweight="bold", pad=10, color="#FFFFFF")
ax2.set_xlabel("Global Step (Cumulative)", fontsize=10)
ax2.set_ylabel("Classification Precision (%)", fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.15, color="#FFFFFF")
legend2 = ax2.legend(facecolor='#181820', edgecolor='#2A2A35')
matt.setp(legend2.get_texts(), color='#E0E0E6')

matt.tight_layout()
matt.show()