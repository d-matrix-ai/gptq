import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mltools import dmx, numerical


def _quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

try:
    import quant_cuda
except:
    print('CUDA extension not installed.')
    

class INTQuantizer(nn.Module):
    def __init__(self, shape=1):
        super(INTQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        self.block_size = 1
        self.test = False

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = _quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return _quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
    
    
class SBFPQuantizer(nn.Module):
    def __init__(self, fmt="sbfp", num_bits=4, block_size=128, sebias=7, test=False):
        super().__init__()
        self.fmt = fmt
        self.block_size = block_size
        self.num_bits = num_bits
        self.test = test
        self.sebias = sebias
        
    def quantize_blocks_sbfp(self, blocks, block_dim=-1, num_bits=4, dequantize=True):
        # find maximum for every block
        max_vals = blocks.abs().amax(block_dim, keepdim=True)

        # compute scaling factors for INT4 range mapping
        scaling_factors = max_vals / (2 ** (num_bits - 1) - 1)    # 7 for INT4
        # if the entire block is zeroes, keep them as zeroes (to avoid division by zero).
        scaling_factors[max_vals == 0] = 1
        
        # map original to INT4 range
        blocks_q = blocks / scaling_factors
        # quantize by rounding, then clip to remove any values > 7
        blocks_q = blocks_q.round_()#.clamp_(min=-8, max=7)  # clamping is unnecessary after rounding

        if dequantize:
            # map the quantized values to the original range
            blocks_dq = blocks_q * scaling_factors
            return blocks_dq
        else:
            return blocks_q, scaling_factors
            
    def quantize_weights_sbfp(self, W, layer=None):
        self.num_rows, self.num_cols = W.shape
        assert self.num_cols % self.block_size == 0   # we should be receiving already padded weights (in GPTQ init function)
        num_blocks = int(self.num_cols / self.block_size)
        split_weights = W.reshape(self.num_rows, num_blocks, self.block_size)   # [768, 2304] --> [768, 72, 32]
        
        layer.block_size = self.block_size  # <-- this is coming from model dmx config file
        layer.dtype = W.dtype
        layer.use_loop = 1
        layer.bfp_block_size = 128
        layer.num_bits = self.num_bits
        layer.bfp_num_bits = 8   # 8 bits for BFP16, 4 bits for BFP12
        layer.num_blocks = num_blocks
        layer.test = self.test

        if self.test:
            # GPTQ block: 128, SBFP block: 64    
            # process (128, 48, 64) chunk out of (768, 48, 64)
            weights_sbfp = self.quantize_blocks_sbfp(split_weights, block_dim=-1, num_bits=self.num_bits, dequantize=True)
            return weights_sbfp.reshape(self.num_rows, self.num_cols)

        else:
            split_weights_int4, scaling_factors = self.quantize_blocks_sbfp(split_weights, block_dim=-1, num_bits=self.num_bits, dequantize=False)
            layer.already_quantized = True
            return split_weights_int4, scaling_factors
            split_weights_int4 = split_weights_int4.to(W.dtype)
            split_weights_int4 = split_weights_int4.transpose(0, 1)   # [768, 72, 32] --> [72, 768, 32]
            scaling_factors = scaling_factors.permute(1, 2, 0).unsqueeze(0).to(W.dtype)   # [768, 72, 1] --> [1, 72, 768] --> [72, 1, 768] --> [1, 72, 1, 768]
            layer.split_weight_shape = split_weights_int4.shape
            return split_weights_int4, scaling_factors

    def configure(self, bits, perchannel=True, sym=False, mse=False):
        pass
        
    def find_params(self, *args, **kwargs):
        # all dummy values below, not used
        self.maxq = None
        self.scale = None
        self.zero = None

    def quantize(self, W, layer=None):
        #breakpoint()
        return self.quantize_weights_sbfp(W, layer=layer)

    def enabled(self):
        return True

    def ready(self):
        return True

class DMXQuantizer(nn.Module):
    def __init__(self, fmt="bfp", block_size=128, sebias=7):
        super().__init__()
        self.fmt = fmt
        self.block_size = block_size
        self.sebias = sebias
        self.test = False

    def configure(self, bits, perchannel=True, sym=False, mse=False):
        if self.fmt == "bfp":
            self.format = numerical.BlockFloatingPoint(precision=bits, block_size=self.block_size, block_dim=-1)
        elif self.fmt == "sbfp" and bits == 4:
            self.format = numerical.ScaledBlockFloatingPoint(
                block_format=numerical.Format.from_shorthand("XP[4,0](CSN)"),
                scaler_format=numerical.FloatingPoint(mantissa=4, exponent=4, bias=self.sebias, flush_subnormal=True, unsigned=True, rounding="nearest"),
                block_size=self.block_size,
                block_dim=-1,
            )
        else:
            raise ValueError(
                f"unsupported precision {bits} for d-Matrix numerical format {self.fmt}"
            )
        self.cast_to = dmx.CastTo(format=self.format)

    def find_params(self, *args, **kwargs):
        # all dummy values below, not used
        self.maxq = None
        self.scale = None
        self.zero = None

    def quantize(self, x):
        if self.ready:
            # if x.shape[-1] == 1:
            #     # NOTE: ugly fix due to GPTQ's unsqueeze() before quantize() call
            #     return self.cast_to(x.squeeze(-1).float()).unsqueeze(-1)
            # else:
            return self.cast_to(x.float())
        return x

    def enabled(self):
        return True

    def ready(self):
        return True


try:
    import quant_cuda
except:
    print("CUDA extension not installed.")

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module):
    def __init__(self, infeatures, outfeatures):
        super().__init__()
        self.register_buffer("zeros", torch.zeros((outfeatures, 1)))
        self.register_buffer("scales", torch.zeros((outfeatures, 1)))
        self.register_buffer("bias", torch.zeros(outfeatures))
        self.register_buffer(
            "qweight",
            torch.zeros((infeatures // 1024 * 96, outfeatures), dtype=torch.int),
        )

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(
            torch.int
        )
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 1024 * 96, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            x = x.float()
            quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError("Only supports a single token currently.")


def make_quant3(module, names, name=""):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(module, attr, Quant3Linear(tmp.in_features, tmp.out_features))
    for name1, child in module.named_children():
        make_quant3(child, names, name + "." + name1 if name != "" else name1)
