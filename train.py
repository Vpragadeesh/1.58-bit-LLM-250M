import os
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
import torch.nn as nn
import bitsandbytes as bnb
from model import BitNetLM

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

console = Console()

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
CONTEXT_WINDOW = 512
BATCH_SIZE = 2             # Stable on RTX 2050 (3.68 GB VRAM)
ACCUMULATION_STEPS = 16    # Effective batch = 2 * 16 = 32
MAX_STEPS = 25_000
LR = 3e-4
DATA_PATH = "data/processed/train.bin"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 1000    # Save every 1000 steps

def get_batch(data, batch_size, block_size, device):
    """Fetch a random batch from the memory-mapped dataset."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

def save_checkpoint(model, optimizer, step, loss, path):
    """Save training checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path, device):
    """Load training checkpoint from disk. Returns the step to resume from."""
    if not os.path.exists(path):
        return 0
    console.print(f"[bold yellow]Resuming from checkpoint:[/bold yellow] {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    console.print(f"  → Step [bold cyan]{ckpt['step']}[/bold cyan], Loss [bold cyan]{ckpt['loss']:.4f}[/bold cyan]")
    return ckpt['step']

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(DATA_PATH):
        console.print(f"[bold red]Error:[/bold red] Dataset not found at {DATA_PATH}. Please run prepare.py first.")
        return

    # ── Memory-map data once ──
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')

    # ── Build model ──
    model = BitNetLM(
        vocab_size=50257,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=2730,
        max_seq_len=CONTEXT_WINDOW
    )
    model.to(device)
    model.train()

    param_count = sum(p.numel() for p in model.parameters())

    # ── Optimizer ──
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Mixed precision scaler (uses fp16 activations for ~2× speed) ──
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Resume from checkpoint ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pt")
    start_step = load_checkpoint(model, optimizer, latest_ckpt, device)

    # ── CUDA optimizations ──
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Display config ──
    total_vram = ""
    if device.type == 'cuda':
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        total_vram = f" / {total_vram_gb:.1f} GB total"
    
    config_info = (
        f"• [bold cyan]Device:[/bold cyan] {device}{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}\n"
        f"• [bold cyan]VRAM:[/bold cyan] {total_vram.strip(' / ') if total_vram else 'N/A'}\n"
        f"• [bold cyan]Model Parameters:[/bold cyan] {param_count / 1e6:.2f} M\n"
        f"• [bold cyan]Context Window:[/bold cyan] {CONTEXT_WINDOW} tokens\n"
        f"• [bold cyan]Batch Size:[/bold cyan] {BATCH_SIZE} (effective {BATCH_SIZE * ACCUMULATION_STEPS} with grad accum)\n"
        f"• [bold cyan]Precision:[/bold cyan] 1.58-bit weights + fp16 mixed precision\n"
        f"• [bold cyan]Optimizer:[/bold cyan] 8-Bit AdamW (lr={LR})\n"
        f"• [bold cyan]Checkpoint:[/bold cyan] Every {CHECKPOINT_EVERY} steps → {CHECKPOINT_DIR}/\n"
        f"• [bold cyan]Dataset Tokens:[/bold cyan] {len(data):,}\n"
        f"• [bold cyan]Resuming from step:[/bold cyan] {start_step}"
    )
    console.print(Panel(config_info, title="[bold green]⚡ BitNet 250M Training Configuration[/bold green]", expand=False))

    # ── Progress bar ──
    optimizer.zero_grad()
    start_time = time.time()
    tokens_processed = 0

    progress = Progress(
        TextColumn("[bold green]Training[/bold green]"),
        BarColumn(bar_width=40, style="black on white", complete_style="bold green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("• ETA:"),
        TimeRemainingColumn(),
        TextColumn("• Loss: [bold cyan]{task.fields[loss]}"),
        TextColumn("• Speed: [bold magenta]{task.fields[tps]}"),
        TextColumn("• VRAM: [bold yellow]{task.fields[vram]}"),
        TextColumn("• Done at: [bold white]{task.fields[eta_ist]}"),
        console=console,
    )

    task_id = progress.add_task(
        "train",
        total=MAX_STEPS,
        completed=start_step,
        loss="N/A",
        tps="0 t/s",
        vram="N/A",
        eta_ist="calculating..."
    )

    current_loss = "N/A"
    current_tps = "0 t/s"
    current_vram = "N/A"
    current_eta_ist = "calculating..."
    raw_loss = 0.0
    IST = timezone(timedelta(hours=5, minutes=30))
    training_start = time.time()

    with progress:
        for step in range(start_step, MAX_STEPS):
            X, Y = get_batch(data, BATCH_SIZE, CONTEXT_WINDOW, device)

            # Mixed precision forward pass
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(X, use_checkpointing=True)
                loss = criterion(logits.view(-1, 50257), Y.view(-1))
                loss_scaled = loss / ACCUMULATION_STEPS

            raw_loss = loss.item()

            # Mixed precision backward pass
            scaler.scale(loss_scaled).backward()

            tokens_processed += X.numel()

            # Optimizer step after gradient accumulation
            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Metrics
                elapsed_time = time.time() - start_time
                tps = tokens_processed / max(elapsed_time, 1e-9)
                current_tps = f"{tps:,.0f} t/s"
                current_loss = f"{raw_loss:.4f}"

                if device.type == 'cuda':
                    vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    current_vram = f"{vram_alloc:.2f} GB"
                else:
                    current_vram = "N/A"

                # Estimate completion time in IST
                total_elapsed = time.time() - training_start
                steps_done = step + 1 - start_step
                if steps_done > 0:
                    secs_per_step = total_elapsed / steps_done
                    remaining_steps = MAX_STEPS - (step + 1)
                    remaining_secs = secs_per_step * remaining_steps
                    finish_time = datetime.now(IST) + timedelta(seconds=remaining_secs)
                    current_eta_ist = finish_time.strftime("%I:%M %p IST")

                start_time = time.time()
                tokens_processed = 0

            # Save checkpoint
            if (step + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, step + 1, raw_loss, latest_ckpt)
                # Also save a numbered checkpoint
                numbered_path = os.path.join(CHECKPOINT_DIR, f"step_{step + 1}.pt")
                save_checkpoint(model, optimizer, step + 1, raw_loss, numbered_path)
                console.print(f"\n[bold green]✓ Checkpoint saved[/bold green] at step {step + 1} (loss: {raw_loss:.4f})")

            progress.update(
                task_id,
                advance=1,
                loss=current_loss,
                tps=current_tps,
                vram=current_vram,
                eta_ist=current_eta_ist
            )

    # Final checkpoint
    save_checkpoint(model, optimizer, MAX_STEPS, raw_loss, latest_ckpt)
    console.print(f"\n[bold green]✓ Training complete![/bold green] Final checkpoint saved.")

if __name__ == "__main__":
    main()
