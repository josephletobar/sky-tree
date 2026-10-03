import base64
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

PROMPT = """These images are drone footage frames in chronological order.
Write one dense, factual summary of roughly 50-150 words covering the scene,
terrain, buildings, roads, visible people and vehicles, their spatial relationships,
and changes across the frames. Describe motion only when supported by the sequence.
Avoid repeating details shared across frames. Do not invent identities, intentions,
locations, or details too small to see; mention uncertainty where relevant.
If only one image is provided, summarize that frame without inferring motion.
"""

MODEL = "hf.co/google/gemma-4-31B-it-qat-q4_0-gguf:latest"


def summarize_images(image_paths, prompt=PROMPT):
    images = [base64.b64encode(image if isinstance(image, bytes) else Path(image).read_bytes()).decode()
              for image in image_paths]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "think": False,
        "keep_alive": os.getenv("SKYTREE_KEEP_ALIVE", "0"),
        "options": {"num_ctx": 8192, "num_predict": 512, "temperature": 0.2},
    }
    request = Request("http://localhost:11434/api/chat",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        return json.load(response)["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python summarize_images.py frame1.jpg [frame2.jpg ...]")
    print(summarize_images(sys.argv[1:]))
