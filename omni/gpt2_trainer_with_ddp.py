import os
import time
import math
import torch
from contextlib import nullcontext
from tqdm import tqdm
from logger import get_logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

logger = get_logger(__name__)

seed_offset = 0
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]

ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print('starting ddp on ', device, ddp_local_rank, ddp_world_size)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

def get_ctx(device_type):
    ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
    return ctx


def get_lr(it):
    warmup_iters = 2000
    lr_decay_iters = 10000
    min_lr = 6e-5
    learning_rate = 6e-4

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, ctx, eval_batches):
    model.eval()
    losses = torch.zeros(len(eval_batches))
    for k, (X, Y, tasks) in enumerate(eval_batches):
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


def train(model,
          get_batch,
          out_dir,
          steps=1000,
          batch_size=64,
          block_size=1024,
          grad_accum_steps=16,
          eval_interval = 200,
          eval_steps=100):
    
    print("moving model to", device)
    model.to(device)

    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps //= ddp_world_size

    print(f'Training with {steps} steps, {batch_size} batch size, {block_size} block size')

    os.makedirs(out_dir, exist_ok=True)

    device_type = 'cuda' if 'cuda' in device else 'cpu'

    grad_clip = 1.0
    ctx = get_ctx(device_type)

    tokens_per_iter = grad_accum_steps * batch_size * block_size * ddp_world_size
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")
    print("NUM TOTAL TOKENS:", (tokens_per_iter * steps)/(10**9), "Billion")

    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = model.configure_optimizers(1e-1, get_lr(0), (0.9, 0.95), device_type)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    t0 = time.time()
    local_iter_num = 0

    eval_batches = [get_batch('val', batch_size=batch_size, block_size=block_size, device=device) for i in range(eval_steps)]
    X, Y, _ = get_batch('train', block_size=block_size, batch_size=batch_size, device=device)

    all_losses = {}

    raw_model = model.module if ddp else model
    
    if master_process:
        pbar = tqdm(total=steps)

    for iter_num in range(steps):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(iter_num)

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss(model, ctx, eval_batches)
            logger.info(f"Validation loss: {iter_num}, {losses}")
            all_losses['val'] = losses
            model_fname = f"{out_dir}/gpt_{iter_num}.pt"
            torch.save({"model": raw_model.state_dict(), "config": raw_model.config}, model_fname)

        for micro_step in range(grad_accum_steps):
            if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / grad_accum_steps
            X, Y, _ = get_batch('train', block_size=block_size, batch_size=batch_size, device=device)
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * grad_accum_steps
            all_losses['train'] = lossf
            loss_string = f"train: {all_losses['train']:.4f} val: {all_losses['val']:.4f}"
            pbar.set_description(loss_string)
            pbar.update(iter_num - pbar.n) 
        
        iter_num += 1
        local_iter_num += 1

    model_fname = f"{out_dir}/gpt_last.pt"
    torch.save({"model": raw_model.state_dict(), "config": raw_model.config}, model_fname)

    if ddp:
        destroy_process_group()

    return model_fname

def dummy_get_batch(split, block_size, batch_size, device):
    X = torch.zeros(batch_size, block_size, dtype=torch.long).to(device)
    Y = torch.ones(batch_size, block_size, dtype=torch.long).to(device)
    return X, Y


if __name__ == '__main__':
    from gpt2_model import get_model
    model = get_model(
        n_layer=4,
        n_head=4,
        n_embd=256,
        vocab_size=3072,
        block_size=1024
    )

    train(
        model,
        get_batch=dummy_get_batch,
        out_dir='out',
        steps=3000,
        block_size=1024,
        eval_interval=5,
        eval_steps=4,
        batch_size=64
    )
