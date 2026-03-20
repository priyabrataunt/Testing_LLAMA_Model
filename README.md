# Testing LLAMA Model

A simple interactive chatbot implementation using the Llama 2 7B Chat model with Hugging Face Transformers.

## Description

This project provides a command-line interface for interacting with the Llama 2 7B Chat model. It loads the model locally and allows you to have a conversation with the AI assistant through a simple terminal interface.

## Features

- Interactive chat interface with Llama 2 7B Chat model
- Automatic GPU/CPU device detection
- Optimized memory usage for large language models
- Customizable generation parameters (temperature, max tokens, etc.)
- Proper Llama 2 chat prompt formatting

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for better performance)
- At least 16GB RAM (32GB recommended for GPU usage)
- Sufficient disk space for the model (~13GB)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/priyabrataunt/Testing_LLAMA_Model.git
cd Testing_LLAMA_Model
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Llama 2 7B Chat model:

You can download the model from Hugging Face:
```bash
# Using Hugging Face CLI
huggingface-cli login
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./models/llama-2-7b-chat
```

Or update the `MODEL_PATH` in `chat.py` to use a Hugging Face model ID directly:
```python
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
```

Note: You may need to request access to the Llama 2 model on Hugging Face first.

## Usage

Run the chat interface:
```bash
python chat.py
```

The script will:
1. Detect whether to use GPU or CPU
2. Load the tokenizer and model
3. Start an interactive chat session

Type your messages and press Enter to chat with the model. Type `quit` or `exit` to end the session.

### Example Session

```
Using device: cuda
Loading tokenizer...
Loading model... (this may take a minute)

🦙 Llama-2-7b-chat is ready! Type 'quit' to exit.

You: What is machine learning?

Llama: Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data...

You: quit
```

## Configuration

You can customize the chat behavior by modifying parameters in the `chat()` function:

- `max_new_tokens`: Maximum number of tokens to generate (default: 512)
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative) (default: 0.7)
- `top_p`: Nucleus sampling parameter (default: 0.9)
- `repetition_penalty`: Penalty for repeating tokens (default: 1.1)

You can also customize the system prompt to change the assistant's behavior:
```python
response = chat("Your question here", system_prompt="You are a Python programming expert.")
```

## Project Structure

```
Testing_LLAMA_Model/
├── chat.py           # Main chat interface script
├── README.md         # This file
├── requirements.txt  # Python dependencies
└── models/          # Directory for storing model files (not tracked in git)
```

## Technical Details

- Uses PyTorch for tensor operations and model inference
- Leverages Hugging Face Transformers library for model loading and generation
- Implements proper Llama 2 chat prompt formatting with system and user messages
- Supports automatic device mapping for efficient GPU/CPU usage
- Uses fp16 precision on GPU for faster inference and reduced memory usage

## Troubleshooting

### Out of Memory Errors
- Try running on CPU instead of GPU
- Reduce `max_new_tokens` parameter
- Close other applications to free up memory

### Model Not Found
- Ensure the model path is correct in `chat.py`
- Verify the model has been downloaded completely
- Check that you have access to the Llama 2 model on Hugging Face

### Slow Performance
- Use a GPU if available (CUDA-compatible)
- Ensure PyTorch is installed with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

This project is for testing and educational purposes. Please comply with the Llama 2 license terms when using the model.

## Acknowledgments

- Meta AI for the Llama 2 model
- Hugging Face for the Transformers library
