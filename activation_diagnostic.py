import argparse
import csv
import json
from pathlib import Path

import torch


@torch.no_grad()
def collect_all_channel_activations(model, batches, top_k: int = 20):
    blocks = model.model.transformer.blocks
    locations = {"q_proj_input": {}, "post_block": {}}

    def record(location, index, hidden):
        values = hidden.detach().float().abs().reshape(-1, hidden.shape[-1])
        total = values.sum(0, dtype=torch.float64).cpu()
        maximum = values.max(0).values.cpu()
        state = locations[location].setdefault(
            index,
            {"sum": torch.zeros_like(total), "max": torch.zeros_like(maximum), "count": 0},
        )
        state["sum"] += total
        state["max"] = torch.maximum(state["max"], maximum)
        state["count"] += values.shape[0]

    def capture_q_input(index):
        def hook(_, inputs):
            record("q_proj_input", index, inputs[0])

        return hook

    def capture_block_output(index):
        def hook(_, __, output):
            record("post_block", index, output[0] if isinstance(output, tuple) else output)

        return hook

    handles = []
    for index, block in enumerate(blocks):
        handles.append(block.q_proj.register_forward_pre_hook(capture_q_input(index)))
        handles.append(block.register_forward_hook(capture_block_output(index)))
    try:
        for batch in batches:
            model(batch[0] if isinstance(batch, (tuple, list)) else batch)
    finally:
        for handle in handles:
            handle.remove()

    if any(len(states) != len(blocks) for states in locations.values()):
        raise ValueError("Activation diagnostic received no samples")

    def summarize(states):
        ordered = [states[index] for index in range(len(blocks))]
        sums = torch.stack([state["sum"] for state in ordered])
        maxima = torch.stack([state["max"] for state in ordered])
        counts = torch.tensor([state["count"] for state in ordered], dtype=torch.float64)
        block_means = sums / counts[:, None]
        means = sums.sum(0) / counts.sum()
        max_values = maxima.max(0).values

        def rank(values):
            return sorted(range(values.numel()), key=lambda channel: (-values[channel].item(), channel))

        ranking = rank(means)
        return {
            "mean_abs": means.tolist(),
            "max_abs": max_values.tolist(),
            "ranking": ranking,
            "top_channels": ranking[:top_k],
            "by_block": {
                index: {
                    "mean_abs": block_means[index].tolist(),
                    "max_abs": maxima[index].tolist(),
                    "ranking": rank(block_means[index]),
                }
                for index in range(len(blocks))
            },
        }

    profile = {location: summarize(states) for location, states in locations.items()}
    profile["channel_count"] = len(profile["q_proj_input"]["mean_abs"])
    return profile


def write_all_channel_profile(profile, output: Path, metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                **metadata,
                "primary_metric": "q_proj_input mean(abs), averaged over samples, tokens, and blocks",
                "profile": profile,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("location", "block", "channel", "mean_abs", "max_abs", "mean_abs_rank"),
        )
        writer.writeheader()
        for location in ("q_proj_input", "post_block"):
            for block, stats in profile[location]["by_block"].items():
                ranks = {channel: rank for rank, channel in enumerate(stats["ranking"], start=1)}
                for channel, (mean_abs, max_abs) in enumerate(zip(stats["mean_abs"], stats["max_abs"])):
                    writer.writerow(
                        {
                            "location": location,
                            "block": block,
                            "channel": channel,
                            "mean_abs": mean_abs,
                            "max_abs": max_abs,
                            "mean_abs_rank": ranks[channel],
                        }
                    )


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
    parser.add_argument("--all-channels", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
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
    if args.all_channels:
        profile = collect_all_channel_activations(model, batches, args.top_k)
        write_all_channel_profile(
            profile,
            Path(args.output),
            {"model": args.model, "samples": args.samples, "seed": args.seed},
        )
        return

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
