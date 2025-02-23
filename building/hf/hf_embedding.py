import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    # Load the pretrained GPT-2 model and tokenizer from Hugging Face
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # My thoughts on this are in the notes I wrote down.
    # tokenizer encodes and decodes text into tokens and back to text.
    # im unsure how we actuall process batches of text.
    # do we have all tensors in our batch to share the same sequence?
    # how do we even process a batch of text?
    
    model.eval()  # set to evaluation mode

    # Prepare a sample input text
    sample_text = "Hello, world!"
    input_ids = tokenizer.encode(sample_text, return_tensors="pt")

    # Isolate the token embedding layer (wte) and run a forward pass using it
    with torch.no_grad():
        # The token embedding layer is accessible via model.transformer.wte
        embeddings = model.transformer.wte(input_ids)

    print("Sample text:", sample_text)
    print("Input IDs:", input_ids)
    print("Embeddings shape:", embeddings.shape)
    print("Embeddings:", embeddings)

if __name__ == "__main__":
    main()
