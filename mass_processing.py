import os
import re
import csv
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

# ============================================================
# QWEN VIDEO BACKEND
# ============================================================

os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"

from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# NETWORK DIRECTORY
# ------------------------------------------------------------
#
# Example:
#
# \\192.168.1.100\video
#
# or
#
# \\SERVER\VideoArchive
#

WATCH_DIRECTORY = Path(
    r"\\SERVER\Video"
)

# ------------------------------------------------------------
# LOCAL DIRECTORIES
# ------------------------------------------------------------

OUTPUT_DIRECTORY = PROJECT_DIR / "output"
CACHE_DIRECTORY = PROJECT_DIR / "cache"
LOG_DIRECTORY = PROJECT_DIR / "logs"

PROCESSED_FILE = PROJECT_DIR / "processed_files.txt"


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

SAMPLE_FPS = 1.0

MAX_NEW_TOKENS = 1024


# ------------------------------------------------------------
# DIRECTORY SCANNING
# ------------------------------------------------------------

SCAN_INTERVAL_SECONDS = 30

# File must have the same size during two checks.
FILE_STABILITY_SECONDS = 10


# ------------------------------------------------------------
# VIDEO EXTENSIONS
# ------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".mpeg",
    ".mpg",
    ".webm",
}


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Проанализируй предоставленное видео.

На видео могут быть постоянно или периодически наложены служебные
надписи системы видеонаблюдения.

Необходимо выполнить две задачи:

1. Определить служебные данные из надписей на видео.
2. Составить хронологию происходящих событий.

СЛУЖЕБНЫЕ ДАННЫЕ

Попытайся определить:

- дату видеозаписи;
- время, отображаемое на видео;
- имя или номер камеры;
- название производственного поста, участка или рабочей зоны.

Если какое-либо значение отсутствует или невозможно уверенно
распознать, используй null.

Не придумывай значения.

ХРОНОЛОГИЯ

Опиши значимые события строго в порядке их возникновения.

Для каждого события обязательно укажи:

- время начала;
- время окончания;
- описание события.

Время события указывай относительно начала видео в формате:

HH:MM:SS

Например:

00:00:12
00:00:26

Не создавай отдельное событие для каждого кадра.

Если одно действие продолжается длительное время, объединяй его
в один интервал.

Фиксируй в первую очередь:

- появление человека;
- уход человека;
- перемещение человека;
- появление или перемещение техники;
- работу с изделием;
- работу с инструментом;
- начало производственной операции;
- окончание производственной операции;
- загрузку;
- разгрузку;
- перемещение материалов;
- простои;
- отсутствие активности;
- другие заметные изменения состояния рабочей зоны.

Ответ верни ТОЛЬКО в формате JSON.

Никакого текста до JSON и после JSON.

Структура должна быть строго следующей:

{
    "metadata": {
        "date": null,
        "time": null,
        "camera": null,
        "post": null
    },
    "events": [
        {
            "start": "00:00:00",
            "end": "00:00:10",
            "description": "Описание события"
        }
    ]
}

Если события отсутствуют:

"events": []

Если служебные данные не удалось определить,
оставь соответствующие поля null.
"""


# ============================================================
# GLOBAL MODEL OBJECTS
# ============================================================

model = None
processor = None


# ============================================================
# LOGGING
# ============================================================

def log(message):

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    text = f"[{now}] {message}"

    print(
        text,
        flush=True,
    )

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        LOG_DIRECTORY
        / f"{datetime.now():%Y-%m-%d}.log"
    )

    with open(
        log_file,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            text + "\n"
        )


# ============================================================
# FILE NAME SANITIZER
# ============================================================

def sanitize_filename(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        value,
    )

    value = re.sub(
        r"\s+",
        "_",
        value,
    )

    value = value.strip(
        "._ "
    )

    if not value:
        return None

    return value[:80]


# ============================================================
# PROCESSED FILE DATABASE
# ============================================================

def load_processed_files():

    if not PROCESSED_FILE.exists():
        return set()

    processed = set()

    with open(
        PROCESSED_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            value = line.strip()

            if value:
                processed.add(value)

    return processed


def get_processed_key(video_path):

    try:

        relative = video_path.relative_to(
            WATCH_DIRECTORY
        )

        return str(relative).lower()

    except Exception:

        return str(
            video_path
        ).lower()


def mark_as_processed(video_path):

    key = get_processed_key(
        video_path
    )

    with open(
        PROCESSED_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            key + "\n"
        )


# ============================================================
# FILE STABILITY CHECK
# ============================================================

def is_file_ready(path):

    try:

        size1 = path.stat().st_size

        if size1 == 0:
            return False

        time.sleep(
            FILE_STABILITY_SECONDS
        )

        size2 = path.stat().st_size

        if size1 != size2:

            log(
                f"File is still being copied: "
                f"{path.name}"
            )

            return False

        # Attempt opening the file.
        with open(
            path,
            "rb",
        ):
            pass

        return True

    except Exception as e:

        log(
            f"File is not ready: "
            f"{path.name}: {e}"
        )

        return False


# ============================================================
# VIDEO SEARCH
# ============================================================

def find_video_files():

    files = []

    if not WATCH_DIRECTORY.exists():

        log(
            f"Network directory unavailable: "
            f"{WATCH_DIRECTORY}"
        )

        return files

    try:

        for path in WATCH_DIRECTORY.rglob("*"):

            try:

                if not path.is_file():
                    continue

                if (
                    path.suffix.lower()
                    not in VIDEO_EXTENSIONS
                ):
                    continue

                files.append(
                    path
                )

            except Exception:
                continue

    except Exception as e:

        log(
            f"Directory scanning error: {e}"
        )

    # Oldest files first.

    try:

        files.sort(
            key=lambda p: p.stat().st_mtime
        )

    except Exception:
        pass

    return files


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global model
    global processor

    log(
        f"Loading model: {MODEL_ID}"
    )

    start = time.perf_counter()

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIRECTORY,
        )
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIRECTORY,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
    )

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    log(
        f"Model loaded in "
        f"{elapsed:.2f} seconds"
    )

    if torch.cuda.is_available():

        log(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code fences if model added them.

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:

        return json.loads(
            text
        )

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if (
            start >= 0
            and end > start
        ):

            candidate = text[
                start:end + 1
            ]

            return json.loads(
                candidate
            )

        raise


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path):

    log(
        f"Analyzing: {video_path}"
    )

    total_start = time.perf_counter()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(
                        video_path
                    ),
                    "fps": SAMPLE_FPS,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }
    ]

    # --------------------------------------------------------
    # Video decoding / vision preprocessing
    # --------------------------------------------------------

    stage_start = time.perf_counter()

    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )

    vision_time = (
        time.perf_counter()
        - stage_start
    )

    log(
        f"Vision preprocessing: "
        f"{vision_time:.2f} s"
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    stage_start = time.perf_counter()

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(
        model.device
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    input_time = (
        time.perf_counter()
        - stage_start
    )

    log(
        f"Input preparation: "
        f"{input_time:.2f} s"
    )

    if "input_ids" in inputs:

        log(
            f"Input tokens: "
            f"{inputs['input_ids'].shape[-1]}"
        )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stage_start = time.perf_counter()

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_time = (
        time.perf_counter()
        - stage_start
    )

    generated_ids_trimmed = [
        output_ids[
            len(input_ids):
        ]
        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    generated_tokens = (
        generated_ids_trimmed[0]
        .shape[0]
    )

    log(
        f"Generation: "
        f"{generation_time:.2f} s"
    )

    log(
        f"Generated tokens: "
        f"{generated_tokens}"
    )

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    output_text = (
        processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    log(
        f"Total video analysis time: "
        f"{total_time:.2f} s"
    )

    if torch.cuda.is_available():

        peak = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        log(
            f"Peak GPU memory: "
            f"{peak:.2f} GB"
        )

    # Keep raw answer for troubleshooting.

    raw_directory = (
        OUTPUT_DIRECTORY
        / "raw"
    )

    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_name = (
        video_path.stem
        + ".txt"
    )

    with open(
        raw_directory / raw_name,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            output_text
        )

    result = extract_json(
        output_text
    )

    result["_processing"] = {
        "vision_seconds": round(
            vision_time,
            3,
        ),
        "input_seconds": round(
            input_time,
            3,
        ),
        "generation_seconds": round(
            generation_time,
            3,
        ),
        "total_seconds": round(
            total_time,
            3,
        ),
        "generated_tokens": int(
            generated_tokens
        ),
    }

    return result


# ============================================================
# OUTPUT CSV NAME
# ============================================================

def choose_csv_file(metadata):

    date = sanitize_filename(
        metadata.get(
            "date"
        )
    )

    camera = sanitize_filename(
        metadata.get(
            "camera"
        )
    )

    post = sanitize_filename(
        metadata.get(
            "post"
        )
    )

    # We consider metadata sufficient only when
    # camera and post were recognized.

    if camera and post:

        parts = []

        if date:
            parts.append(
                date
            )

        parts.append(
            camera
        )

        parts.append(
            post
        )

        filename = (
            "_".join(parts)
            + ".csv"
        )

        return (
            OUTPUT_DIRECTORY
            / filename
        )

    return (
        OUTPUT_DIRECTORY
        / "common.csv"
    )


# ============================================================
# CSV WRITING
# ============================================================

CSV_COLUMNS = [
    "video_file",
    "video_path",
    "recording_date",
    "recording_time",
    "camera",
    "post",
    "event_start",
    "event_end",
    "event_description",
    "analysis_date",
    "analysis_seconds",
    "generation_seconds",
    "generated_tokens",
]


def save_result(
    video_path,
    result,
):

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = (
        result.get(
            "metadata"
        )
        or {}
    )

    events = (
        result.get(
            "events"
        )
        or []
    )

    processing = (
        result.get(
            "_processing"
        )
        or {}
    )

    csv_path = choose_csv_file(
        metadata
    )

    file_exists = (
        csv_path.exists()
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=CSV_COLUMNS,
            delimiter=";",
        )

        if not file_exists:

            writer.writeheader()

        # Even an empty video gets one row.

        if not events:

            events = [
                {
                    "start": "",
                    "end": "",
                    "description":
                        "Значимые события не обнаружены",
                }
            ]

        for event in events:

            writer.writerow(
                {
                    "video_file":
                        video_path.name,

                    "video_path":
                        str(video_path),

                    "recording_date":
                        metadata.get(
                            "date"
                        )
                        or "",

                    "recording_time":
                        metadata.get(
                            "time"
                        )
                        or "",

                    "camera":
                        metadata.get(
                            "camera"
                        )
                        or "",

                    "post":
                        metadata.get(
                            "post"
                        )
                        or "",

                    "event_start":
                        event.get(
                            "start"
                        )
                        or "",

                    "event_end":
                        event.get(
                            "end"
                        )
                        or "",

                    "event_description":
                        event.get(
                            "description"
                        )
                        or "",

                    "analysis_date":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "analysis_seconds":
                        processing.get(
                            "total_seconds",
                            "",
                        ),

                    "generation_seconds":
                        processing.get(
                            "generation_seconds",
                            "",
                        ),

                    "generated_tokens":
                        processing.get(
                            "generated_tokens",
                            "",
                        ),
                }
            )

    log(
        f"CSV saved: {csv_path}"
    )

    return csv_path


# ============================================================
# PROCESS SINGLE FILE
# ============================================================

def process_file(
    video_path,
    processed_files,
):

    key = get_processed_key(
        video_path
    )

    if key in processed_files:
        return

    log(
        "=" * 70
    )

    log(
        f"New video detected: "
        f"{video_path.name}"
    )

    if not is_file_ready(
        video_path
    ):

        return

    try:

        result = analyze_video(
            video_path
        )

        metadata = (
            result.get(
                "metadata"
            )
            or {}
        )

        log(
            "Metadata: "
            f"date={metadata.get('date')}, "
            f"time={metadata.get('time')}, "
            f"camera={metadata.get('camera')}, "
            f"post={metadata.get('post')}"
        )

        events = (
            result.get(
                "events"
            )
            or []
        )

        log(
            f"Events detected: "
            f"{len(events)}"
        )

        save_result(
            video_path,
            result,
        )

        # IMPORTANT:
        # mark file only AFTER successful CSV saving.

        mark_as_processed(
            video_path
        )

        processed_files.add(
            key
        )

        log(
            f"Processing completed: "
            f"{video_path.name}"
        )

    except Exception as e:

        log(
            f"ERROR processing "
            f"{video_path.name}: {e}"
        )

        log(
            traceback.format_exc()
        )

        # File is intentionally NOT marked as processed,
        # therefore it will be retried next time.


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    log(
        "VideoAnalyzer started"
    )

    log(
        f"Watch directory: "
        f"{WATCH_DIRECTORY}"
    )

    processed_files = (
        load_processed_files()
    )

    log(
        f"Previously processed files: "
        f"{len(processed_files)}"
    )

    load_model()

    log(
        "Waiting for video files..."
    )

    while True:

        try:

            video_files = (
                find_video_files()
            )

            new_files = [
                path
                for path in video_files
                if get_processed_key(path)
                not in processed_files
            ]

            if new_files:

                log(
                    f"New files found: "
                    f"{len(new_files)}"
                )

                for video_path in new_files:

                    process_file(
                        video_path,
                        processed_files,
                    )

            else:

                log(
                    "No new files."
                )

        except KeyboardInterrupt:

            log(
                "Stopped by user."
            )

            break

        except Exception as e:

            log(
                f"Main loop error: {e}"
            )

            log(
                traceback.format_exc()
            )

        time.sleep(
            SCAN_INTERVAL_SECONDS
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()