import openai
import time
import sys
from PIL import Image
import io
import base64
import prompt

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
def generate_image_caption_detailed(image, engine: str = "gpt-4o") -> str:
    FIXED_PROMPT = prompt.prompt_for_caption

    image_url = open_clip_compatible_encode_image(image)

    chat_message = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FIXED_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model=engine,
            messages=chat_message,
            temperature=0.0,
            max_tokens=2048,
            timeout=60
        )
        caption = response.choices[0].message.content.strip()
        return caption

    except AuthenticationError:
        print(" API failure, check OpenAI Key！", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Description generation failure：{str(e)}", file=sys.stderr)
        raise


def open_clip_compatible_encode_image(image_path: str) -> str:
    with Image.open(image_path).convert("RGB") as img:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()

    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(60))
def openai_completion(prompt, engine="gpt-4.1", max_tokens=700, temperature=0):
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )
    try:
        resp = client.chat.completions.create(
            model=engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=300,
            # request_timeout=10,
            temperature=temperature,
            stop=["\n\n", "<|endoftext|>"]
        )

        return resp.choices[0].message.content

    except (AuthenticationError, APIError, Timeout) as e:
        print(f"Request failure：{str(e)}")
    except Exception as e:
        print(f"Error：{str(e)}")




