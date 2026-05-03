# Instead of using AutoModelForCausalLM, you can use AutoModelForMultimodalLM to process audio. To use it, make sure to install the following packages:

# pip install -U transformers torch torchvision librosa accelerate

# You can then load the model with the code below:

from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_ID = "/data/tmp/yongqiang/nfs/auto_model_deployment/gemma-4-hf-original/gemma-4-E2B-it/"

# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID, 
    dtype="auto", 
    device_map="auto"
)

# Once the model is loaded, you can start generating output by directly referencing the audio URL in the prompt:

# Prompt - add audio before text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "assets/gemma4_audio_test_chunk0_30s.wav"},
            {"type": "text", "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three."},
        ]
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