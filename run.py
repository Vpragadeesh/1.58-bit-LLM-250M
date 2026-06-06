import os
import torch
import tiktoken
from model import BitNetLM

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/latest.pt"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50
CONTEXT_WINDOW = 512

def load_model(checkpoint_path, device):
    """Load the trained BitNet model from a checkpoint."""
    model = BitNetLM(
        vocab_size=50257,
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=2730,
        max_seq_len=CONTEXT_WINDOW
    )

    if not os.path.exists(checkpoint_path):
        console.print(f"[bold red]Error:[/bold red] Checkpoint not found at {checkpoint_path}")
        console.print("Please train the model first using train.py")
        return None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    return model, ckpt.get('step', '?'), ckpt.get('loss', '?')

@torch.no_grad()
def generate(model, tokenizer, prompt_text, device, max_new_tokens=200, temperature=0.8, top_k=50):
    """Generate text autoregressively from a prompt."""
    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt_text, allowed_special={'<|endoftext|>'})
    
    # Truncate if prompt is too long
    if len(token_ids) >= CONTEXT_WINDOW:
        token_ids = token_ids[-(CONTEXT_WINDOW - 1):]
    
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated_tokens = []

    for _ in range(max_new_tokens):
        # Crop to context window
        idx_cond = input_ids[:, -CONTEXT_WINDOW:]

        # Forward pass
        logits = model(idx_cond, use_checkpointing=False)

        # Get logits for the last token
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, float('-inf')), logits)

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=1)
        generated_tokens.append(next_token.item())

        # Stop on end-of-text token
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(generated_tokens)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    console.print("\n[bold cyan]Loading model...[/bold cyan]")
    result = load_model(CHECKPOINT_PATH, device)
    if result is None:
        return
    model, step, loss = result

    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    param_count = sum(p.numel() for p in model.parameters())

    # Display info
    info = (
        f"• [bold cyan]Model:[/bold cyan] BitNet 250M (1.58-bit)\n"
        f"• [bold cyan]Parameters:[/bold cyan] {param_count / 1e6:.2f} M\n"
        f"• [bold cyan]Device:[/bold cyan] {device}\n"
        f"• [bold cyan]Checkpoint:[/bold cyan] Step {step} (loss: {f'{loss:.4f}' if isinstance(loss, float) else loss})\n"
        f"• [bold cyan]Temperature:[/bold cyan] {TEMPERATURE}\n"
        f"• [bold cyan]Top-K:[/bold cyan] {TOP_K}\n"
        f"• [bold cyan]Max Tokens:[/bold cyan] {MAX_NEW_TOKENS}"
    )
    console.print(Panel(info, title="[bold green]⚡ BitNet 250M Inference[/bold green]", expand=False))

    console.print("[dim]Type your prompt and press Enter. Type 'quit' or 'exit' to stop.[/dim]")
    console.print("[dim]Commands: /temp <val>  /topk <val>  /tokens <val>[/dim]\n")

    # Interactive loop
    while True:
        try:
            prompt = Prompt.ask("[bold green]>>>[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Goodbye![/bold yellow]")
            break

        if not prompt.strip():
            continue

        if prompt.strip().lower() in ('quit', 'exit'):
            console.print("[bold yellow]Goodbye![/bold yellow]")
            break

        # Handle settings commands
        if prompt.startswith('/temp '):
            try:
                TEMP = float(prompt.split()[1])
                console.print(f"[dim]Temperature set to {TEMP}[/dim]")
                globals()['TEMPERATURE'] = TEMP
                continue
            except (ValueError, IndexError):
                console.print("[red]Usage: /temp 0.8[/red]")
                continue

        if prompt.startswith('/topk '):
            try:
                TK = int(prompt.split()[1])
                console.print(f"[dim]Top-K set to {TK}[/dim]")
                globals()['TOP_K'] = TK
                continue
            except (ValueError, IndexError):
                console.print("[red]Usage: /topk 50[/red]")
                continue

        if prompt.startswith('/tokens '):
            try:
                MT = int(prompt.split()[1])
                console.print(f"[dim]Max tokens set to {MT}[/dim]")
                globals()['MAX_NEW_TOKENS'] = MT
                continue
            except (ValueError, IndexError):
                console.print("[red]Usage: /tokens 200[/red]")
                continue

        # Generate
        console.print()
        with console.status("[bold magenta]Generating...[/bold magenta]"):
            output = generate(
                model, tokenizer, prompt, device,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K
            )

        console.print(Panel(output, title="[bold magenta]Generated Output[/bold magenta]", expand=False))
        console.print()

if __name__ == "__main__":
    main()
