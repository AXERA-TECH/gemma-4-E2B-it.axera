from pathlib import Path

from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "google/gemma-4-E2B-it"
VIDEO_PATH = Path(__file__).resolve().parent.parent / "assets" / "red-panda-openai.mp4"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)

# Prompt - add local video before text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": str(VIDEO_PATH)},
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Process input
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse output
processor.parse_response(response)

print(response)
print(processor.parse_response(response))

