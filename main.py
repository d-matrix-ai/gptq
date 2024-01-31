import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="OPT or BLOOM model name (opt_125m, bloom_560m, etc)")
parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
parser.add_argument("--percdamp", type=float, default=0.01, help="Percent of the average Hessian diagonal to use for dampening.")
parser.add_argument("--format", type=str, choices=["bfp", "sbfp", "int"], default="sbfp", help="Quantization format")
parser.add_argument("--bfp_blocksize", type=int, default=128, help="BFP block size (128 for BFP12, 64 for BFP16)")
parser.add_argument("--sbfp_blocksize", type=int, default=None, help="SBFP block size")
parser.add_argument("--gptq_blocksize", type=int, default=128, help="GPTQ block size")
parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
parser.add_argument( "--wbits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8, 9, 16], help="#bits to use for quantization; use 16 for evaluating base model.")
parser.add_argument("--sebias", type=int, default=7, choices=[7, 8, 9, 10, 11], help="for SBFP, uFP scaler's exponent bias")
parser.add_argument( "--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row.")
parser.add_argument("--check", action="store_true", help="Whether to compute perplexity during benchmarking for verification.")
parser.add_argument("--test", action="store_true", help="Test simple SBFP weight quantization (no splitting)")
parser.add_argument("--qkv", action="store_true", default=False, help="use linear layers with SBFP kernel, only in the QKV linear layer")
parser.add_argument("--mlp1", action="store_true", default=False, help="use linear layers with SBFP kernel, only in the first MLP layer")
parser.add_argument("--mlp2", action="store_true", default=False, help="use linear layers with SBFP kernel, only in the second MLP layer")
parser.add_argument("--all", action="store_true", default=False, help="use linear layers with SBFP kernel, in all linear layers")
parser.add_argument("--fp32_calibration", action="store_true", default=False, help="compute GPTQ Hessians using FP32 weights/inputs")
parser.add_argument("--gpu", type=str, default=None, help="GPU ID to use if specified")
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
import torch
import torch.nn as nn

from datautils import *
from gptq import *
from quant import *
from functools import partial

from mltools import dmx
from mlreferences import (
    squad_bigbird_large, 
    squad_deberta_v3_large, 
    squad_bert_large, 
    squad_bert_base, 
    
    opt_125m, 
    opt_350m, 
    opt_1b3, 
    opt_2b7, 
    opt_6b7, 
    opt_13b, 
    opt_30b, 
    opt_66b, 
    
    distilgpt2,
    gpt2,  
    gpt2_medium, 
    gpt2_large, 
    gpt2_xl, 
    
    bloom_560m, 
    bloom_1b7, 
    bloom_3b, 
    bloom_7b1, 
    
    llama_7b, 
    
    squad_t5_small, 
    squad_t5_base, 
    squad_t5_large,
    squad_t5_11b,
    
    lenet5,
    lenet_512_512,
    )


DEBUG = 1
device = torch.device('cuda:0')
LAYERS = [nn.Conv2d, nn.Linear]
DMX_LAYERS = (dmx.nn.Linear, dmx.nn.HFTransformersConv1D)
    
    
def find_layers(module, layers=DMX_LAYERS, sbfp=False, name=''):
    if (type(module) in layers) and ((module.layer_type is not None and 'SBFP' in module.layer_type) or not sbfp):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, sbfp=sbfp, name=name + '.' + name1 if name != '' else name1))
    return res


@torch.no_grad()
def opt_sequential(model, dataloader, device, seqlen=None, quantizer_cls=None, exp_str=None, args=None):
    print(f'\n\nCalibrating {args.model} on {args.nsamples} samples from {args.dataset} dataset')
    
    use_cache = model.body.config.use_cache
    model.body.config.use_cache = False
    
    if 'opt' in args.model:
        layers = model.body.model.decoder.layers
        
        model.body.model.decoder.embed_tokens = model.body.model.decoder.embed_tokens.to(device)
        model.body.model.decoder.embed_positions = model.body.model.decoder.embed_positions.to(device)
        if hasattr(model.body.model.decoder, "project_out") and model.body.model.decoder.project_out:
            model.body.model.decoder.project_out = model.body.model.decoder.project_out.to(device)
        if hasattr(model.body.model.decoder, "project_in") and model.body.model.decoder.project_in:
            model.body.model.decoder.project_in = model.body.model.decoder.project_in.to(device)
            
        cache = {"i": 0, "attention_mask": None}
        
    elif 'bloom' in args.model:
        layers = model.body.transformer.h
        
        model.body.transformer.word_embeddings = model.body.transformer.word_embeddings.to(device)
        model.body.transformer.word_embeddings_layernorm = (model.body.transformer.word_embeddings_layernorm.to(device))
        
        cache = {"i": 0, "attention_mask": None, "alibi": None}
        
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype  # why?
    
    # collect inputs to the first decoder block
    inputs = torch.zeros((len(dataloader), seqlen, model.body.config.hidden_size), dtype=dtype, device=device)      # OPT-125m:  (128, 2048, 768)
    

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inputs[cache["i"]] = inp  # cache["i"] is just an integer index to dataloader, this is a recording mechanism: why not use input hook?
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if 'bloom' in args.model:
                cache["alibi"] = kwargs["alibi"]
            raise ValueError  # an ugly way of making the forward stop after layer[0]

    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass  # an ugly way of making the forward stop after layer[0]
        
    # by now we should have decoder inputs from all data examples in the calibration set
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    if 'opt' in args.model:
        model.body.model.decoder.embed_tokens = model.body.model.decoder.embed_tokens.cpu()
        model.body.model.decoder.embed_positions = model.body.model.decoder.embed_positions.cpu()
        if hasattr(model.body.model.decoder, "project_out") and model.body.model.decoder.project_out:
            model.body.model.decoder.project_out = model.body.model.decoder.project_out.cpu()
        if hasattr(model.body.model.decoder, "project_in") and model.body.model.decoder.project_in:
            model.body.model.decoder.project_in = model.body.model.decoder.project_in.cpu()
        
    elif 'bloom' in args.model:
        model.body.transformer.word_embeddings = model.body.transformer.word_embeddings.cpu()
        model.body.transformer.word_embeddings_layernorm = (model.body.transformer.word_embeddings_layernorm.cpu())
        
        
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inputs)   # outs will become next decoder block inputs
    quant_str = f'INT{args.wbits}' if args.format == 'int' else f'{args.format}_{quantizer_cls().block_size}_{args.wbits}bit'
    print(f'\n\nQuantizing {args.model} weights to {quant_str} using GPTQ algorithm\n\n')

    for i in range(len(layers)):      # going through decoder layers
        layer = layers[i].to(device)  # move layers to GPU one at a time
        subset = find_layers(layer, sbfp=args.format == 'sbfp')   # subset is a dictionary holding 6 Linear modules in the layer
        if len(subset) == 0:
            print(f'No layers selected for quantization in block {i}')
            continue
        
        gptq = {}                     # dictionary to hold gptq objects
        # gptq object holds each linear layer module, layer weights W, num_rows/cols, Hessian, and num_samples
        for name in subset:
            quantizer = quantizer_cls()
            gptq[name] = GPTQ(subset[name], quantizer=quantizer)      # GPTQ object with a layer
            if args.test and isinstance(quantizer, SBFPQuantizer):
                gptq[name].layer.test = True
            if args.fp32_calibration:
                gptq[name].layer.fp32_calibration = True
            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)  # why not specify as args to Quantizer constructor?

        # compute Hessians for each layer (done in gptq.add_batch function)
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(len(dataloader)):
            # this outs list is not used (only to produce Hessians)
            if 'opt' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"])[0]
            elif 'bloom' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"], alibi=cache['alibi'])[0]
            
        for h in handles:
            h.remove()
            
        # by now every linear layers in this layer (block) should have a Hessian attached
            
        fp32_cal_str = ' (FP32 calibration)' if args.fp32_calibration else ''
        print(f'\nQuantizing {exp_str} layers in decoder block {i}{fp32_cal_str}')
        for name in subset:
            # disable FP32 calibration mode if it was enabled
            if args.fp32_calibration:
                gptq[name].layer.fp32_calibration = False
                
            weights_before = gptq[name].W.detach().cpu().numpy()
            print(f'\n\t{name} weights before quantization: {list(gptq[name].W.shape)}\n')
            
            gptq[name].fasterquant(blocksize=args.gptq_blocksize, percdamp=args.percdamp, groupsize=args.groupsize)  # GPTQ algo
            
            if not args.test and isinstance(quantizer, SBFPQuantizer):
                weights_after = gptq[name].layer.weight.detach().cpu().numpy().astype(np.float16)
                scaling_factors =  gptq[name].layer.scaling_factors[0].permute(0, 2, 1).cpu().numpy().astype(np.float16)
                dequant_weights = weights_after * scaling_factors
                quant_values_str = f'{weights_before[:6, 0]}\n\t{weights_after[0, 0, :6]}\n\t{dequant_weights[0, 0, :6]}'
                dequant_weights_orig_shape = dequant_weights.transpose(1, 0, 2).reshape(weights_before.shape[1], -1).T[:weights_before.shape[0], :]
            else:
                dequant_weights = gptq[name].layer.weight.detach().cpu().numpy().astype(np.float16)
                quant_values_str = f'{weights_before[:6, 0]}\n\t{dequant_weights[:6, 0]}'
                dequant_weights_orig_shape = dequant_weights
            
            quant_error = 100 * (weights_before - dequant_weights_orig_shape).__abs__().sum() / weights_before.__abs__().sum()
            print(f'\n\t{name} weights after quantization: {list(gptq[name].layer.weight.shape)}\n\n\t{quant_values_str}\n\n\tquant error {quant_error:.2f}%\n')
            gptq[name].free()
            
        print('\n')
            
        # compute output to use as next layer input
        for j in range(len(dataloader)):
            if 'opt' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"])[0]
            elif 'bloom' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"], alibi=cache['alibi'])[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inputs, outs = outs, inputs  # pass this layer output as input to next layer

    model.body.config.use_cache = use_cache


@torch.no_grad()
def opt_eval(model, testenc, device, seqlen=None, quantizer_cls=None, args=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen

    use_cache = model.body.config.use_cache
    model.body.config.use_cache = False
    
    if 'opt' in args.model:
        layers = model.body.model.decoder.layers
        
        model.body.model.decoder.embed_tokens = model.body.model.decoder.embed_tokens.to(device)
        model.body.model.decoder.embed_positions = model.body.model.decoder.embed_positions.to(device)
        if hasattr(model.body.model.decoder, "project_out") and model.body.model.decoder.project_out:
            model.body.model.decoder.project_out = model.body.model.decoder.project_out.to(device)
        if hasattr(model.body.model.decoder, "project_in") and model.body.model.decoder.project_in:
            model.body.model.decoder.project_in = model.body.model.decoder.project_in.to(device)
            
        cache = {"i": 0, "attention_mask": None}
        
    elif 'bloom' in args.model:
        layers = model.body.transformer.h
        
        model.body.transformer.word_embeddings = model.body.transformer.word_embeddings.to(device)
        model.body.transformer.word_embeddings_layernorm = (model.body.transformer.word_embeddings_layernorm.to(device))
        
        cache = {"i": 0, "attention_mask": None, "alibi": None}
    
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros((nsamples, seqlen, model.body.config.hidden_size), dtype=dtype, device=device)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inputs[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if 'bloom' in args.model:
                cache["alibi"] = kwargs["alibi"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
        try:
            model(batch)
        except ValueError:
            pass
        
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    if 'opt' in args.model:
        model.body.model.decoder.embed_tokens = model.body.model.decoder.embed_tokens.cpu()
        model.body.model.decoder.embed_positions = model.body.model.decoder.embed_positions.cpu()
        if hasattr(model.body.model.decoder, "project_out") and model.body.model.decoder.project_out:
            model.body.model.decoder.project_out = model.body.model.decoder.project_out.cpu()
        if hasattr(model.body.model.decoder, "project_in") and model.body.model.decoder.project_in:
            model.body.model.decoder.project_in = model.body.model.decoder.project_in.cpu()
    elif 'bloom' in args.model:
        model.body.transformer.word_embeddings = model.body.transformer.word_embeddings.cpu()
        model.body.transformer.word_embeddings_layernorm = (model.body.transformer.word_embeddings_layernorm.cpu())
        
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inputs)

    for i in range(len(layers)):
        layer = layers[i].to(device)

        if args.nearest and args.wbits < 16:
            print(f'\nQuantizing {args.model} block {i} weights to {args.format} {args.wbits} bits (RTN algorithm)')
            subset = find_layers(layer, sbfp=args.format == 'sbfp')
            
            for name in subset:
                print(f'\t{name}')
                quantizer = quantizer_cls()
                quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)    # can also pass: quantizer.scale, quantizer.zero, quantizer.maxq

        for j in range(nsamples):
            if 'opt' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"])[0]
            elif 'bloom' in args.model:
                outs[j] = layer(inputs[j].unsqueeze(0), attention_mask=cache["attention_mask"], alibi=cache['alibi'])[0]
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inputs, outs = outs, inputs

    if 'opt' in args.model:
        if model.body.model.decoder.final_layer_norm is not None:
            model.body.model.decoder.final_layer_norm = model.body.model.decoder.final_layer_norm.to(device)
        if model.body.model.decoder.project_out is not None:
            model.body.model.decoder.project_out = model.body.model.decoder.project_out.to(device)
    elif 'bloom' in args.model:
        model.body.transformer.ln_f = model.body.transformer.ln_f.to(device)
        
    model.body.lm_head = model.body.lm_head.to(device)

    testenc = testenc.to(device)
    nlls = []
    for i in range(nsamples):
        hidden_states = inputs[i].unsqueeze(0)
        
        if 'opt' in args.model:
            if model.body.model.decoder.final_layer_norm is not None:
                hidden_states = model.body.model.decoder.final_layer_norm(hidden_states)
            if model.body.model.decoder.project_out is not None:
                hidden_states = model.body.model.decoder.project_out(hidden_states)
        elif 'bloom' in args.model:
            hidden_states = model.body.transformer.ln_f(hidden_states)
            
        lm_logits = model.body.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
        
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    model.body.config.use_cache = use_cache
    return ppl.item()


if __name__ == "__main__":
    dmx.aware(patch_hf_transformers=True)

    results = {}
    exp_str = ''
    
    if args.model is not None:
        models = [args.model]
    else:
        models = ['opt_125m', 'opt_350m', 'opt_1b3', 'opt_2b7']
        models = ['bloom_560m', 'bloom_1b7']
    
    if args.format == 'sbfp':
        if args.sbfp_blocksize is None:
            blocksizes = [256, 512, 1024]
        else:
            blocksizes = [args.sbfp_blocksize]
    else:
        blocksizes = [args.bfp_blocksize]
    
    for model in models:
        args.model = model
        model_results = {}
        
        for blocksize in blocksizes:
            
            if args.format == "int":
                Quantizer = INTQuantizer
            elif args.format == "bfp":
                Quantizer = partial(DMXQuantizer, fmt=args.format, block_size=args.bfp_blocksize)
            elif args.format == "sbfp":
                Quantizer = partial(SBFPQuantizer, fmt=args.format, num_bits=args.wbits, block_size=blocksize, sebias=args.sebias, test=args.test)

            wl = eval(model)()
            
            config = eval(f'wl.dmx_configs.BASELINE')
            
            if args.mlp1:
                if args.format == 'sbfp':
                    config = eval(f'wl.dmx_configs.BASELINE_SBFP_{blocksize}_MLP1')
                exp_str = 'mlp1'
            elif args.mlp2:
                if args.format == 'sbfp':
                    config = eval(f'wl.dmx_configs.BASELINE_SBFP_{blocksize}_MLP2')
                exp_str = 'mlp2'
            elif args.qkv:
                if args.format == 'sbfp':
                    config = eval(f'wl.dmx_configs.BASELINE_SBFP_{blocksize}_QKV')
                exp_str = 'qkv'
            elif args.all:
                if args.format == 'sbfp':
                    config = eval(f'wl.dmx_configs.BASELINE_SBFP_{blocksize}')
                exp_str = 'all'
            else:
                exp_str = 'no'
                
            print(f'\n\nTransforming model to {config}')            
            wl.model.transform(config)

            if 'opt' in args.model:
                tokenizer_model = 'facebook/opt-125m'
            elif 'bloom' in args.model:
                tokenizer_model = 'bigscience/bloom-560m'
                    
            if args.wbits < 16 and not args.nearest:
                print(f'\n\nGetting {args.nsamples} samples from {args.dataset} dataset for calibration')
                dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=tokenizer_model, seqlen=wl.max_seq_len)
                quantizers = opt_sequential(wl.model, dataloader, device, seqlen=wl.max_seq_len, quantizer_cls=Quantizer, exp_str=exp_str, args=args)

            print(f'\n\nGetting full {args.dataset} dataloader')
            dataloader, testloader = get_loaders(args.dataset, seed=args.seed, model=tokenizer_model, seqlen=wl.max_seq_len)
                
            print(f'\n\nEvaluating {model} on {args.dataset} dataset')
            ppl = opt_eval(wl.model, testloader, device, seqlen=wl.max_seq_len, quantizer_cls=Quantizer, args=args) 
            print(f"\n\nPerpexity: {ppl:.2f}\n\n") 
            
            model_results[blocksize] = ppl
            print(f'\n\nModel {model}: quantizing {exp_str} layers using GPTQ and {args.format}\n')
            print(model_results, '\n\n')
            for bs, ppl in model_results.items():
                print(f'\nModel {model}: perplexity for {args.format} blocksize {bs:>4}:  {ppl:5.2f}') 
            print('\n\n')
    
        results[model] = model_results
        print(results, '\n\n')
        for mod, res in results.items():
            print(f'\nModel {mod}')
            for bs, ppl in res.items():
                print(f'\tPerplexity for {args.format} blocksize {bs:>4}:  {ppl:5.2f}')
    print('\n\n\n')
