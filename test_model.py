import os
import sys
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = os.path.join(os.getcwd(), "cache")


def print_separator():
    print("=" * 70)


def main():
    print_separator()
    print("VideoAnalyzer - Qwen2.5-VL model test")
    print_separator()

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {total_vram / 1024**3:.2f} GB")

    print_separator()
    print(f"Model: {MODEL_ID}")
    print(f"Cache: {CACHE_DIR}")
    print_separator()

    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )

    print("Processor loaded.")

    print()
    print("Loading model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model.eval()

    print("Model loaded.")

    print_separator()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"GPU memory allocated: {allocated:.2f} GB")
    print(f"GPU memory reserved:  {reserved:.2f} GB")

    print_separator()
    print("Running text inference...")
    print_separator()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Ответь одним предложением: модель успешно работает?",
                }
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print()
    print("MODEL RESPONSE:")
    print(response)
    print()

    print_separator()
    print("TEST COMPLETED SUCCESSFULLY")
    print_separator()


if __name__ == "__main__":
    main()