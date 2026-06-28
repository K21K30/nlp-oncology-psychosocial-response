"""
check_setup.py — Environment check before building the validated-generation pipeline.

Verifies which models and interfaces are available, so the call_generator / call_judge_a /
call_judge_b functions can be wired correctly. Prints a clear report. Reads .env for the
OpenAI key but never prints the key itself.

Run:  python check_setup.py
"""

import os
import sys

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)

# --- Python ---
print(f"\nPython: {sys.version.split()[0]}")

# --- packages ---
print("\n[1] Required packages:")
for pkg in ["ollama", "openai", "dotenv", "tqdm", "pandas", "sklearn"]:
    try:
        mod = __import__(pkg if pkg != "dotenv" else "dotenv")
        ver = getattr(mod, "__version__", "?")
        print(f"    OK   {pkg:10s} {ver}")
    except Exception as e:
        print(f"    MISS {pkg:10s} -> NOT installed ({e})")

# --- Ollama server + models ---
print("\n[2] Ollama (local generator + local judge):")
try:
    import ollama
    models = [m.model for m in ollama.list().models]
    print(f"    Ollama server reachable. Models installed: {len(models)}")
    for want in ["gemma2:27b", "qwen2.5:32b"]:
        mark = "OK  " if want in models else "MISS"
        print(f"    {mark} {want}")
    other = [m for m in models if m not in ("gemma2:27b", "qwen2.5:32b")]
    if other:
        print(f"    (other models present: {', '.join(other)})")
except Exception as e:
    print(f"    ERROR reaching Ollama: {type(e).__name__}: {e}")
    print("    -> Make sure the Ollama app is running.")

# --- OpenAI key + live ping ---
print("\n[3] OpenAI (cloud judge gpt-4o-mini):")
try:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("    MISS OPENAI_API_KEY not found in .env")
    else:
        print(f"    OK   OPENAI_API_KEY found (starts '{key[:7]}...', length {len(key)})")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "reply with: ok"}],
                max_tokens=5,
            )
            print(f"    OK   live ping -> '{r.choices[0].message.content.strip()}'")
        except Exception as e:
            print(f"    ERROR live ping failed: {type(e).__name__}: {e}")
except Exception as e:
    print(f"    ERROR loading .env: {e}")

# --- data files present ---
print("\n[4] Current data files:")
from pathlib import Path
for p in ["data/raw_dataset.jsonl", "data/pilot_audited.csv", ".env", ".gitignore"]:
    print(f"    {'OK  ' if Path(p).exists() else 'miss'} {p}")

print("\n" + "=" * 60)
print("SUMMARY for Claude — copy this whole output back:")
print("  - generator: gemma2:27b via Ollama?")
print("  - judge A:   gpt-4o-mini via OpenAI?")
print("  - judge B:   qwen2.5:32b via Ollama?")
print("If all three show OK above, the answer to Claude's question is '1'.")
print("=" * 60)
