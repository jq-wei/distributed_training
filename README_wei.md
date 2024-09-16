Here we build distributed training platform with multiple GPU/NPU. 

At this moment we focus on small model (<=8b) finetuning and distillation. We fix our ultimate target platform by using deepspeed directly, which support fsdp.

We use Llama-factory as a starting point.




# Install Llama-Factory 
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Use `deepspeed== 0.14.4`

# Steps to run
## CMD for full sft of llama3 and Phi

`FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml > terminal_log.txt`
where in `llama3_full_sft_ds3.yaml` we can set the model/data path, deepspeed config (with/without offload). 

If we use offload for the optimizer state to cpu, then we need to use cpu_adam.cu, which have the following 'well-known' bug `AttributeError: 'DeepSpeedCPUAdam' object has no attribute 'ds_opt_adam'`. (TODO) 

## Someone suggest the following to append to ds config to perform fsdp on zero3

"pipeline": {
  "enabled": true,
  "stages": 2
  },
"tensor_parallel": {
  "enabled": true,
  "tp_size": 2
  }

## FSDP: the following cmd is for running fsdp using accelerate. This also works fine. 

`CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file examples/accelerate/fsdp_config_2_proc.yaml \
    src/train.py examples/extras/fsdp_full_sft/llama3_sft.yaml
`

## Some statistics on training

For meta-llama/Llama-2-7b-hf training using Adam (Peak vRAM) is 49.48 GB (float16/bfloat16). So DDP will not work.  

For microsoft/Phi-3.5-mini-instruct, it needs 28.47 GB

## FSDP

I don't think FSDP is implemented using native torch in LlamaFactory. [my issue](https://github.com/hiyouga/LLaMA-Factory/issues/5441)
But since FSDP FULL_SHARD maps to DeepSpeed ZeRO stage 3, we are fine with Deepspeed. 

## some note, 

The full training in full_sft was for phi3.5 mini.
For Llama 3.1 8b, seq_len = 1024, we have OOM. It seems more offload is needed.  (TODO)

## terminologies

data parallel
model parallel
tensor parallel
pipeline parallel
FSDP