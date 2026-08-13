import argparse
import json
from pathlib import Path

import torch


@torch.no_grad()
def collect_channel_activations(model, batches, channel: int, block_indices: tuple[int, ...]):
    blocks = model.model.transformer.blocks
    if any(index < 0 or index >= len(blocks) for index in block_indices):
        raise ValueError("Diagnostic block index is outside the model")

    sums = {index: 0.0 for index in block_indices}
    counts = {index: 0 for index in block_indices}

    def capture(index):
        def hook(_, __, output):
            hidden = output[0] if isinstance(output, tuple) else output
            values = hidden[..., channel].float().abs()
            sums[index] += values.sum().item()
            counts[index] += values.numel()

        return hook

    handles = [blocks[index].register_forward_hook(capture(index)) for index in block_indices]
    try:
        for batch in batches:
            model(batch[0] if isinstance(batch, (tuple, list)) else batch)
    finally:
        for handle in handles:
            handle.remove()

    if any(count == 0 for count in counts.values()):
        raise ValueError("Activation diagnostic received no samples")
    return {index: sums[index] / counts[index] for index in block_indices}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--channel", type=int, default=3848)
    parser.add_argument("--blocks", default="0,15,31")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache_dir", default="llm_weights")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from lib.data import get_loaders
    from model import LLaDAModelLM

    model = LLaDAModelLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    model.seqlen = model.config.max_sequence_length
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.samples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    input_device = model.model.transformer.wte.weight.device
    batches = (batch[0].to(input_device) for batch in dataloader)
    block_indices = tuple(int(index) for index in args.blocks.split(","))
    activations = collect_channel_activations(model, batches, args.channel, block_indices)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": args.model,
                "channel": args.channel,
                "blocks": list(block_indices),
                "samples": args.samples,
                "seed": args.seed,
                "mean_absolute_activation": activations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
