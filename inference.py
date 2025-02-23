from PIL import Image
from unsloth import FastVisionModel
from transformers import TextStreamer

from data import INSTRUCTION

def inference_on_image(image_path: str):
    """
    Inference on image
    :param image_path: image path
    """

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="lora_model",
        load_in_4bit=True,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)  # Enable for inference!

    image = Image.open(image_path)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION.format(brand_name="رامین")}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")


    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                       use_cache=True, temperature=1.5, min_p=0.1)


if __name__ == "__main__":
    inference_on_image("image.jpg")