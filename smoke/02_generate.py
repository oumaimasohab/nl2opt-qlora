import statistics

from mlx_lm import generate, load, stream_generate

MODEL_ID = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
MAX_TOKENS = 1024
N_RUNS = 3

SYSTEM_PROMPT = """You are an expert in operations research and Python.
Given an optimization word problem, write a complete, runnable Python script in PuLP that:
1. Defines decision variables with correct bounds and types (continuous or integer).
2. Defines the objective function.
3. Adds every constraint stated in the problem, with a comment naming each one.
4. Solves the model with PuLP using the HiGHS solver.
5. Prints the solve status, the optimal objective value, and the value of each variable.
Return only the Python code in a single ```python block. Do not explain."""

# NL4Opt-style LP word problem. Replace with a verbatim item from the NL4Opt dataset if you want exact parity.
PROBLEM = """A bakery makes two types of cakes: chocolate and vanilla. Each chocolate cake requires
2 cups of flour and 3 eggs, and each vanilla cake requires 1.5 cups of flour and 2 eggs.
The bakery has 300 cups of flour and 400 eggs available per week. Because of oven capacity,
at most 120 cakes can be baked per week, and market demand requires that at least 30 vanilla
cakes are made. The profit is $12 per chocolate cake and $9 per vanilla cake.
How many cakes of each type should the bakery make to maximize profit?"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": PROBLEM},
]


def main():
    model, tokenizer = load(MODEL_ID)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    # 1) Human-readable check: verbose=True prints prompt tps, generation tps, peak memory.
    print("=== generate(verbose=True), single run ===")
    generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=True)

    # 2) Programmatic numbers: the last response from stream_generate carries final stats.
    print(f"\n=== stream_generate x{N_RUNS} (run 1 = warm-up, discarded) ===")
    results = []
    for i in range(1, N_RUNS + 1):
        last = None
        for last in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
            pass
        stats = {
            "prompt_tokens": last.prompt_tokens,
            "prompt_tps": last.prompt_tps,
            "generation_tokens": last.generation_tokens,
            "generation_tps": last.generation_tps,
            "peak_memory_gb": last.peak_memory,
        }
        print(
            f"run {i}: prompt {stats['prompt_tokens']} tok @ {stats['prompt_tps']:.1f} tok/s | "
            f"gen {stats['generation_tokens']} tok @ {stats['generation_tps']:.1f} tok/s | "
            f"peak {stats['peak_memory_gb']:.2f} GB"
        )
        if i > 1:
            results.append(stats)

    gen_tps = statistics.mean(r["generation_tps"] for r in results)
    prompt_tps = statistics.mean(r["prompt_tps"] for r in results)
    gen_tokens = statistics.mean(r["generation_tokens"] for r in results)
    peak = max(r["peak_memory_gb"] for r in results)
    print(
        f"\nRecorded (runs 2-{N_RUNS}): generation {gen_tps:.1f} tok/s, prompt {prompt_tps:.1f} tok/s, "
        f"~{gen_tokens:.0f} output tokens, peak memory {peak:.2f} GB"
    )
    print(f"Estimated seconds per sample: {gen_tokens / gen_tps:.1f}s")


if __name__ == "__main__":
    main()