import os

# Принудительно используем TorchCodec для чтения видео.
# Важно: переменная должна быть установлена ДО импорта qwen_vl_utils.
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"

import sys
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CACHE_DIR = os.path.join(os.getcwd(), "cache")
VIDEO_PATH = os.path.join("input", "test.mp4")


def separator():
    print("=" * 70)


def print_gpu_memory(prefix="GPU memory"):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"{prefix} allocated: {allocated:.2f} GB")
    print(f"{prefix} reserved:  {reserved:.2f} GB")


def main():
    separator()
    print("VideoAnalyzer - Qwen2.5-VL video test")
    separator()

    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_vram = torch.cuda.get_device_properties(0).total_memory
    print(f"VRAM: {total_vram / 1024**3:.2f} GB")

    video_path = os.path.abspath(VIDEO_PATH)
    cache_dir = os.path.abspath(CACHE_DIR)

    print(f"Video: {video_path}")
    print(f"Cache: {cache_dir}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found:\n{video_path}"
        )

    os.makedirs(cache_dir, exist_ok=True)

    separator()
    print("Checking TorchCodec...")
    separator()

    try:
        import torchcodec
        print("TorchCodec import: OK")
    except Exception as exc:
        raise RuntimeError(
            "TorchCodec is not available.\n"
            "Install it with:\n"
            "python -m pip install torchcodec"
        ) from exc

    separator()
    print("Loading processor...")
    separator()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
    )

    print("Processor loaded.")

    separator()
    print("Loading model...")
    separator()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model.eval()

    print("Model loaded.")
    print_gpu_memory("After model loading")

    separator()
    print("Preparing video...")
    separator()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": (
                        "Опиши, что происходит на видео. "
                        "Ответь кратко на русском языке."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("Chat template created.")

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    print("Vision data prepared.")

    if image_inputs is None:
        print("Images: 0")
    else:
        print(f"Images: {len(image_inputs)}")

    if video_inputs is None:
        print("Videos: 0")
    else:
        print(f"Videos: {len(video_inputs)}")

    separator()
    print("Creating model inputs...")
    separator()

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    separator()
    print("Input tensors")
    separator()

    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(
                f"{key}: "
                f"shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, "
                f"device={value.device}"
            )
        else:
            print(
                f"{key}: "
                f"type={type(value).__name__}"
            )

    print()
    print_gpu_memory("Before inference")

    separator()
    print("Running inference...")
    separator()

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
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
    )[0].strip()

    separator()
    print("MODEL RESPONSE")
    separator()

    print(response)

    print()
    print_gpu_memory("After inference")

    separator()
    print("VIDEO TEST COMPLETED SUCCESSFULLY")
    separator()


if __name__ == "__main__":
    main()