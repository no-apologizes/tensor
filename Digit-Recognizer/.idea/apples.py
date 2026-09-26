from magicalponies import ffi, lib  # pyright: ignore[reportMissingImports]

class Tensor:
    def __init__(self, b=None, c=None, h=None, w=None, csv_stream_path=None, c_ptr=None):
        # If we're wrapping an existing C ptr, use it directly
        if c_ptr is not None:
            self._c_tensor = c_ptr
            self.shape = (b, c, h, w)
            return

        # Allocate standard new C tensor struct
        self._c_tensor = lib.tensor_create(b, c, h, w) # Pointer to raw C struct
        self.shape = (b, c, h, w)

        if csv_stream_path is not None:
            # Allocate native C integer array for batch target labels
            # Allows C binary to write classification labels that Python can still read like a standard list
            self.labels_out = ffi.new("int[]", b)

            #Python strings are Unicode objects, C funcs only understand null-termed arrays so we convert to raw binary bytes that C can understand
            c_path_bytes = csv_stream_path.encode('utf-8')

            # Open file on C side and hold it there
            lib.tensor_init_csv_stream(c_path_bytes)

    # Matplotlib Python Sequence Protocol
    # Read
    def __len__(self):
            # Calculate logical size to skip over padding
            if self.shape[0] is None:
                return 0
            return self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]
    
    def __getitem__(self, idx):
            size = len(self)
            
            # Handle Python range slicing (e.g., tensor[:batch_log_idx])
            if isinstance(idx, slice):
                start, stop, step = idx.indices(size)
                return [self._c_tensor.data[i] for i in range(start, stop, step)]
    
            # Standard integer indexing
            if idx < 0:
                idx += size
            if idx < 0 or idx >= size:
                raise IndexError("Out of Bounds")
    
            return self._c_tensor.data[idx]

    # Write
    def __setitem__(self, idx, value):
        size = len(self)
        if idx < 0:
            idx += size
        if idx < 0 or idx >= size:
            raise IndexError("Out of Bounds")

        # Write
        self._c_tensor.data[idx] = float(value)

    def load_next_batch(self):
        status = lib.tensor_read_csv_batch(self._c_tensor, self.labels_out, self.shape[0])
        return status == 1 # status because of what the func returns

    def reset_stream(self):
        lib.tensor_reset_csv_stream()

    def __del__(self): # Safety net
        if hasattr(self, '_c_tensor') and self._c_tensor is not None and self._c_tensor != ffi.NULL:
            lib.tensor_free(self._c_tensor)
            # Failsafe close if tensor is destroyed
            try:
                lib.tensor_close_csv_stream()
            except Exception:
                pass

    def zero_data(self):
        lib.tensor_zero_data(self._c_tensor)

    def zero_grad(self):
        lib.tensor_zero_grad(self._c_tensor)

    def copy_data_from(self, src):
        ffi.memmove(
            self._c_tensor.data,
            src._c_tensor.data, 
            src._c_tensor.total_size * ffi.sizeof("float")
        )

    def copy_grad_from(self, src):
        ffi.memmove(
            self._c_tensor.data,
            src._c_tensor.grad, 
            src._c_tensor.total_size * ffi.sizeof("float")
        )

                #ptr ptr ptr float float float float size_t
    def adam_step(self, dW, m, v, lr, beta1, beta2, epsilon, timestep):
        ffi.memmove(self._c_tensor.grad, dW._c_tensor.data, self._c_tensor.total_size * ffi.sizeof("float"))
        lib.tensor_adam_step(self._c_tensor, m._c_tensor, v._c_tensor, lr, beta1, beta2, epsilon, timestep)
                #ptr ptr ptr ptr ptr float size_t
    def muon_step(self, dW, X, XT, Work, lr, ns_steps):
        lib.tensor_muon_step(self._c_tensor, dW._c_tensor, X._c_tensor, XT._c_tensor, Work._c_tensor, lr, ns_steps)

    def transpose_OOP(self, wrt):
        lib.tensor_transpose_OOP(self._c_tensor, wrt._c_tensor)
        return wrt

    def matmul_OOP(self, fallacy, dst): # Matrix multiplication out of place, needs a preallocated C tensor
        lib.tensor_matmul_2d(self._c_tensor, fallacy._c_tensor, dst._c_tensor)
        return dst

    def matmul_backwards(self, dY, dW, XT):
        lib.tensor_matmul_backwards(self._c_tensor, dY._c_tensor, dW._c_tensor, XT._c_tensor)

    def matmul_grad_input(self, dY, dX, WT):
        lib.tensor_matmul_gradient_input(self._c_tensor, dY._c_tensor, dX._c_tensor, WT._c_tensor)

    def conv2d(self, kernel, output):
        lib.tensor_conv2d(self._c_tensor, kernel._c_tensor, output._c_tensor)
        return output

    def im2col(self, kernel_h, kernel_w, output):
        lib.tensor_im2col(self._c_tensor, kernel_h, kernel_w, output._c_tensor)
        return output

    def col2im(self, b, c, h, w, kernel_h, kernel_w, image):
        lib.tensor_col2im(self._c_tensor, b, c, h, w, kernel_h, kernel_w, image._c_tensor)
        return image

    def strided_conv2d_backwards(self, weight, dX):
        lib.tensor_strided_conv2d_backwards(self._c_tensor, weight._c_tensor, dX._c_tensor)
        return dX

    def strided_weight_conv2d_backwards(self, dY, dW):
        lib.tensor_strided_weight_conv2d_backwards(self._c_tensor, dY._c_tensor, dW._c_tensor)
        return dW

    def maxpool2d(self, output, indices):
        lib.tensor_maxpool2d(self._c_tensor, output._c_tensor, indices)
        return output

    def maxpool2d_backwards(self, dX, indices):
        lib.tensor_maxpool2d_backwards(self._c_tensor, dX._c_tensor, indices)
        return dX

    def flatten_copy(self, wrt):
        lib.tensor_flatten_copy(self._c_tensor, wrt._c_tensor)
        return wrt

    def unflatten_copy(self, wrt_grad):
        lib.tensor_unflatten_copy(self._c_tensor, wrt_grad._c_tensor)
        return wrt_grad

def tensor(data_stream_source, dtype=None):

    # If intermediate tensor, flatten directly
    if hasattr(data_stream_source, '_c_tensor'):
        flattened_c_pointer = lib.tensor_flatten_view(data_stream_source._c_tensor)
        flat_tensor = Tensor(64, 1, 1, 5408, c_ptr=flattened_c_pointer)
        flat_tensor._c_tensor = data_stream_source
        if hasattr(data_stream_source, 'labels_out'):
            flat_tensor.labels_out = data_stream_source.labels_out
        return flat_tensor
        
        # Stream directly into a native 1x784 shape
    if hasattr(data_stream_source, 'andie') and isinstance(data_stream_source.andie, str):
        path = data_stream_source.andie
        # stride_w = 784, total_size = 50176
        flat_tensor = Tensor(64, 1, 1, 784, csv_stream_path=path)
        return flat_tensor
        
    raise ValueError("You did something wrong, look in torch.py")

def relu(t):
    lib.tensor_relu(t._c_tensor)
    return t

def relu_backwards(t):
    lib.tensor_relu_backwards(t._c_tensor)
    return t

def gelu(t):
    lib.tensor_gelu(t._c_tensor)
    return t

def gelu_backwards(pre_gelu, grad):
    lib.tensor_gelu_backwards(pre_gelu._c_tensor, grad._c_tensor)
    return grad

def kaiming_init(t, fan_in): # fan_in  is the number of incoming connections feeding into a single neuron or spatial channel in the current layer
    lib.tensor_kaiming_init(t._c_tensor, fan_in)
    return t

def sgd_step(weights, dW, lr): # Learning rate is a float
    ffi.memmove(weights._c_tensor.grad, dW._c_tensor.data, weights._c_tensor.total_size * ffi.sizeof("float"))
    lib.tensor_sgd_step(weights._c_tensor, lr)

def softmax_cross_entropy_loss(hidden, labels):
    accuracy_ptr = ffi.new("float*")
    loss = lib.tensor_softmax_cross_entropy_loss(hidden._c_tensor, labels, accuracy_ptr)
    return loss, accuracy_ptr[0]