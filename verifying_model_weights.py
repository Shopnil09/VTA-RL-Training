from transformers import AutoModelForCausalLM

original = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
trained = AutoModelForCausalLM.from_pretrained("output/checkpoint-5")

# Compare a specific layer's weights
orig_w = original.model.layers[0].self_attn.q_proj.weight
train_w = trained.model.layers[0].self_attn.q_proj.weight

print(f"Weights identical: {(orig_w == train_w).all()}")
print(f"Max difference: {(orig_w - train_w).abs().max()}")