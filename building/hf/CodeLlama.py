# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/CodeLlama-7b-hf",
    device_map="auto",  # Let it automatically choose the best device
    torch_dtype=torch.float16  # Use half precision
)  # Remove the .to("mps") call

sd_hf = model.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)