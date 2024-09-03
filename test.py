import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

AutoConfig.register("praxis", PraxisConfig)
AutoModel.register(PraxisConfig, PraxisModel)
AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)


def test_praxis_model():
    # Initialize configuration
    config = PraxisConfig(
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )

    # Initialize tokenizer (using GPT-2 tokenizer as a placeholder)
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # Initialize model
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Generate dummy input
    input_text = "Hello, world! This is a test."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, return_dict=True)

    # Check outputs
    print("Model Output Shape:", outputs.last_hidden_state.shape)
    if outputs.hidden_states is not None:
        print("Number of layers in output:", len(outputs.hidden_states))
    else:
        print("Hidden states not returned")


def test_praxis_for_causal_lm():
    # Initialize configuration
    config = PraxisConfig(
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )

    # Initialize tokenizer (using GPT-2 tokenizer as a placeholder)
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    # Initialize model
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    # Generate dummy input
    input_text = "Hello, world! This is a test."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Test text generation
    generated = model.generate(input_ids, max_new_tokens=16, num_return_sequences=1)
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)


if __name__ == "__main__":
    print("Testing PraxisModel...")
    test_praxis_model()
    print("\nTesting PraxisForCausalLM...")
    test_praxis_for_causal_lm()
