import openai
import time
import sys
from PIL import Image
import io
import base64

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
from openai import OpenAI, AuthenticationError, APIError, Timeout, RateLimitError

API_KEY = "key_api"
BASE_URL = "https://api.chatanywhere.tech/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


@retry(
    retry=retry_if_exception_type((APIError, Timeout, RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=20),
    reraise=True
)
def generate_with_chain_of_thought(
        sys_prompt: str,
        user_prompt: str,
        image,
        engine: str = "o4-mini",      # "gpt-4o-mini"
        dataset_name: str ="",
        bbox: list = [],
        max_tokens: int = 4096,
        temperature: float = 0,
) -> str:
    if 'genecis' in dataset_name and 'attribute' in dataset_name:
        image_url = open_clip_compatible_encode_image_gencis(image, bbox)
    else:
        image_url = open_clip_compatible_encode_image(image)

    chat_message = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=engine,
            messages=chat_message,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60
        )
        generated_content = response.choices[0].message.content.strip()
        return generated_content

    except AuthenticationError:
        print("!!!", file=sys.stderr)
        raise
    except Exception as e:
        print(f"!!!- {str(e)}", file=sys.stderr)
        raise



def open_clip_compatible_encode_image(image_path: str) -> str:
    with Image.open(image_path).convert("RGB") as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"



from datasets import expand2square
import torch
def open_clip_compatible_encode_image_gencis(image_path: str, bbox) -> str:
    DILATION = 0.7
    PAD_CROP = True

    im = Image.open(image_path).convert('RGB')
    im_width, im_height = im.size

    width = bbox[2]  # Width of bounding box
    height = bbox[3]  # Height of bounding box

    if DILATION:
        orig_left, orig_top = bbox[0], bbox[1]
        left, top = max(0, orig_left - DILATION * width), max(0, orig_top - DILATION * height)
        right, bottom = min(im_width, left + (1 + DILATION) * width), min(im_height, top + (1 + DILATION) * height)
    else:
        left, top = bbox[0], bbox[1]
        right, bottom = bbox[0] + width, bbox[1] + height

    left, top, right, bottom = tensor2num(left), tensor2num(top), tensor2num(right), tensor2num(bottom)
    im = im.crop((left, top, right, bottom))


    buffer = io.BytesIO()
    im.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()

    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def tensor2num(x):
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().item())
    return int(x)





