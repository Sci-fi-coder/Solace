import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY", "").strip()
print("key present:", bool(key), "| length:", len(key))
client = genai.Client(api_key=key)

names = [m.name.removeprefix("models/") for m in client.models.list()]
gem = [n for n in names if n.startswith("gemini")]
print("Gemini models visible to this key:")
for n in gem:
    print("  ", n)

flash = [n for n in gem if "flash" in n and "1.5" not in n and "image" not in n and "live" not in n]
for model in ["gemini-1.5-flash"] + flash[:2]:
    try:
        r = client.models.generate_content(model=model, contents="Reply with the single word: ok")
        print(f"{model}: OK -> {r.text.strip()}")
    except Exception as e:
        print(f"{model}: {type(e).__name__}: {str(e)[:300]}")