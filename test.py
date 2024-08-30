import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from vine import VineConfig, VineModel, VineForCausalLM

AutoConfig.register("vine", VineConfig)
AutoModel.register(VineConfig, VineModel)
AutoModelForCausalLM.register(VineConfig, VineForCausalLM)

def test_vine_model():
    # Initialize configuration
    config = VineConfig(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )

    # Initialize tokenizer (using GPT-2 tokenizer as a placeholder)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Initialize model
    model = VineModel(config)
    model.eval()

    # Generate dummy input
    input_text = "Hello, world! This is a test."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Check outputs
    print("Model Output Shape:", outputs.last_hidden_state.shape)
    if outputs.hidden_states is not None:
        print("Number of layers in output:", len(outputs.hidden_states))
    else:
        print("Hidden states not returned")

def test_vine_for_causal_lm():
    # Initialize configuration
    config = VineConfig(
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )

    # Initialize tokenizer (using GPT-2 tokenizer as a placeholder)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Initialize model
    model = VineForCausalLM(config)
    model.eval()

    # Generate dummy input
    input_text = "Hello, world! This is a test."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Check outputs
    print("Logits Shape:", outputs.logits.shape)
    if outputs.hidden_states is not None:
        print("Number of layers in output:", len(outputs.hidden_states))
    else:
        print("Hidden states not returned")

    # Test text generation
    generated = model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    print("Testing VineModel...")
    test_vine_model()
    print("\nTesting VineForCausalLM...")
    test_vine_for_causal_lm()