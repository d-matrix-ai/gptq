import math
import torch
import torch.nn as nn
import transformers
from mltools import dmx
from quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer, quantizer=None):
        self.layer = layer
        self.quantizer = quantizer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(layer, nn.Linear):
            self.layer.orig_num_cols = W.shape[0]
            
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
            
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.W = W

    def add_batch(self, inp, out):
        # this function computes Hessian for a layer passed as 'self', using a batch of inputs passed as inp
        if isinstance(self.layer, (dmx.nn.Linear, dmx.nn.Conv2d)):
            inp = self.layer.input_cast(inp)      

        if DEBUG:
            self.inp1 = inp
            self.out1 = out
            
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            
        tmp = inp.shape[0]
        
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride,)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
            
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp  # starts from zero, increments by 1 (batch_size=1)
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())   # mean of Hessians across every input sample? why square root?
            
    def fasterquant(self, blocksize=128, percdamp=0.01, groupsize=-1):
        # TODO: make sure weight dim is correct for quantizer
        if isinstance(self.quantizer, DMXQuantizer) or (isinstance(self.quantizer, SBFPQuantizer) and self.quantizer.original):
            assert blocksize % self.quantizer.block_size == 0

        W = self.W.float()
        
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        if isinstance(self.quantizer, SBFPQuantizer) and not self.quantizer.test and not self.quantizer.original:
            scaling_factors_list = []
            split_weights_list = []
            
        # At this point, the layer weights are still unquantized (GPTQ init captured them before nn.py quantized them)
        
        if isinstance(self.quantizer, SBFPQuantizer) and not self.quantizer.original:
            # pad W to multiple of SBFP blocksize
            last_block_size = int(self.layer.orig_num_cols % self.quantizer.block_size)
            if last_block_size != 0:
                last_block_pad = self.quantizer.block_size - last_block_size
                W = F.pad(W.T, (0, last_block_pad), 'constant', 0).T
                print(f'\n\t\tPadding weights to {list(W.shape)}\n')
                
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
                    
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            if isinstance(self.quantizer, DMXQuantizer):
                Q1 = self.quantizer.quantize(W1)
                Err1 = (W1 - Q1).matmul(torch.linalg.inv(Hinv1))
            elif isinstance(self.quantizer, SBFPQuantizer):
                if self.quantizer.test:
                    Q1 = self.quantizer.quantize(W1.T, layer=self.layer).T  #(3072, 128)
                elif self.quantizer.original:
                    Q1 = self.quantizer.quantize(W1.T, layer=self.layer).T  #(3072, 128)
                else:            
                    split_weights_int4, scaling_factors = self.quantizer.quantize(W1.T, layer=self.layer)  # [128, 12, 256], [128, 12, 1]
                    scaling_factors_list.append(scaling_factors)
                    split_weights_list.append(split_weights_int4)
                    Q1 = (split_weights_int4 * scaling_factors).view(blocksize, -1).T    # [128, 12, 256] --> [3072, 128]
                     
                Err1 = (W1 - Q1).matmul(torch.linalg.inv(Hinv1))
            else:
                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    if groupsize != -1 and (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], weight=True)
                            
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        
        if DEBUG:
            print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
            
        
        if isinstance(self.quantizer, SBFPQuantizer):
            if self.quantizer.test or self.quantizer.original:
                # remove padding if any before updating layer weights
                self.layer.weight.data = Q[:self.layer.orig_num_cols, :].reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            else:
                split_weights_int4 = torch.cat(split_weights_list)
                scaling_factors = torch.cat(scaling_factors_list)
                
                split_weights_int4 = split_weights_int4.transpose(0, 1)   # [768, 72, 32] --> [72, 768, 32]
                scaling_factors = scaling_factors.permute(1, 2, 0).unsqueeze(0).to(W.dtype)   # [768, 72, 1] --> [1, 72, 768] --> [72, 1, 768] --> [1, 72, 1, 768]
                
                # the weights here are still padded to be used with the split matmul SBFP algo (matmul output will be trimmed)
                self.layer.weight.data = split_weights_int4
                self.layer.scaling_factors = scaling_factors
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
            
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
