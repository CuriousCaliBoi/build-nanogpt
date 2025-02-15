from transformers import GPT2Model, GPT2Tokenizer

# Initialize the GPT-2 tokenizer and model from Hugging Face
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2")

print("GPT-2 Model Architecture:")
print(gpt2_model)  # Prints a summary of the model's layers and structure

print("\nGPT-2 Model State Dictionary Keys and Weight Shapes:")
for name, param in gpt2_model.named_parameters():
    print(f"{name: <60} shape: {param.shape}")

# Uncomment the following lines if you wish to see the full state dictionary (may produce very verbose output)
# state_dict = gpt2_model.state_dict()
# for key, value in state_dict.items():
#     print(f"{key}: {value.size()}")

