import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "./models/llama-2-7b-chat"  # your local path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Load model & tokenizer ───────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model... (this may take a minute)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto",        # automatically handles GPU/CPU placement
    low_cpu_mem_usage=True,   # important for large models
)

# ── Llama 2 Chat prompt format ───────────────────────────
def format_prompt(user_message: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = "You are a helpful, respectful and honest assistant."
    
    return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""

# ── Generate response ────────────────────────────────────
def chat(user_message: str, system_prompt: str = None,
         max_new_tokens: int = 512, temperature: float = 0.7):
    
    prompt = format_prompt(user_message, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the newly generated tokens (skip the prompt)
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response.strip()

# ── Main loop ────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🦙 Llama-2-7b-chat is ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        print("\nLlama:", chat(user_input))
        print()