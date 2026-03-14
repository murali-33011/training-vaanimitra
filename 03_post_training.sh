#!/bin/bash
# 03_post_training.sh
# Run AFTER training job completes.
# Usage: bash 03_post_training.sh
# Requires: job_info.json in same directory (written by 02_sagemaker_launch.py)

set -e  # exit on any error

echo "=================================================="
echo " Vanimitra Post-Training Pipeline"
echo "=================================================="

# ── Read job info ─────────────────────────────────────────────────────────────
if [ ! -f "job_info.json" ]; then
    echo "ERROR: job_info.json not found. Did 02_sagemaker_launch.py complete?"
    exit 1
fi

BUCKET=$(python3 -c "import json; d=json.load(open('job_info.json')); print(d['bucket'])")
OUTPUT_PATH=$(python3 -c "import json; d=json.load(open('job_info.json')); print(d['output_path'])")
REGION=$(python3 -c "import json; d=json.load(open('job_info.json')); print(d['region'])")

echo "Bucket:      s3://${BUCKET}"
echo "Adapters:    ${OUTPUT_PATH}"
echo "Region:      ${REGION}"
echo ""

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[Step 1/7] Installing dependencies..."
pip install -q transformers==4.40.0 peft==0.10.0 accelerate==0.30.0 torch --quiet
pip install -q sentencepiece protobuf --quiet
echo "           Done."

# ── Download LoRA adapters from S3 ───────────────────────────────────────────
echo "[Step 2/7] Downloading LoRA adapters from S3..."
mkdir -p lora_adapters
# SageMaker wraps output in model.tar.gz
aws s3 cp "${OUTPUT_PATH}" ./model.tar.gz --region "${REGION}"
tar -xzf model.tar.gz -C lora_adapters/
echo "           Done. Contents:"
ls lora_adapters/

# ── Merge LoRA into base model ────────────────────────────────────────────────
echo ""
echo "[Step 3/7] Merging LoRA adapters into base model..."
cat > merge_lora.py << 'PYEOF'
import os, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./lora_adapters"
MERGED_PATH = "./merged_model"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"Base model:   {BASE_MODEL}")
print(f"Adapter path: {ADAPTER_PATH}")
print(f"Output path:  {MERGED_PATH}")

print("Loading base model (fp16)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    token=HF_TOKEN if HF_TOKEN else None,
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN if HF_TOKEN else None,
    trust_remote_code=True,
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_PATH}...")
os.makedirs(MERGED_PATH, exist_ok=True)
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("✅ Merge complete!")
PYEOF

python3 merge_lora.py
echo "           Merge done."

# ── Clone llama.cpp ───────────────────────────────────────────────────────────
echo ""
echo "[Step 4/7] Building llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp --depth=1
fi
cd llama.cpp
pip install -q -r requirements.txt
# Build quantizer binary
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_NATIVE=OFF > /dev/null 2>&1
cmake --build build --config Release -j$(nproc) --target llama-quantize > /dev/null 2>&1
cd ..
echo "           llama.cpp ready."

# ── Convert to GGUF ───────────────────────────────────────────────────────────
echo ""
echo "[Step 5/7] Converting merged model to GGUF (f16)..."
python3 llama.cpp/convert_hf_to_gguf.py \
    ./merged_model \
    --outtype f16 \
    --outfile vanimitra_v1_f16.gguf

echo "           GGUF conversion done."
ls -lh vanimitra_v1_f16.gguf

# ── Quantize to Q4_K_M ────────────────────────────────────────────────────────
echo ""
echo "[Step 6/7] Quantizing to Q4_K_M..."
./llama.cpp/build/bin/llama-quantize \
    vanimitra_v1_f16.gguf \
    vanimitra_v1_q4km.gguf \
    Q4_K_M

echo "           Quantization done."
ls -lh vanimitra_v1_q4km.gguf

# ── Validate on 10 test cases ─────────────────────────────────────────────────
echo ""
echo "[Step 7/7] Validating GGUF on 10 test cases..."
cat > validate_gguf.py << 'PYEOF'
import subprocess, json, sys

GGUF = "./vanimitra_v1_q4km.gguf"
CLI = "./llama.cpp/build/bin/llama-cli"

SYSTEM = """You are a voice command parser for a smartphone assistant.
The user speaks in Hindi, Tamil, or English (or mixed/code-switched).
Output ONLY a single valid JSON object with 'intent' and 'params'.
Valid intents: FLASHLIGHT_ON, FLASHLIGHT_OFF, VOLUME_UP, VOLUME_DOWN,
CALL_CONTACT, SET_ALARM, SEND_WHATSAPP, OPEN_APP, NAVIGATE,
TOGGLE_WIFI, TOGGLE_BLUETOOTH, GO_HOME, GO_BACK, LOCK_SCREEN,
TAKE_SCREENSHOT, UNKNOWN"""

tests = [
    ("टॉर्च जलाओ",               "FLASHLIGHT_ON"),
    ("टॉर्च बंद करो",             "FLASHLIGHT_OFF"),
    ("விளக்கை ஆன் பண்ணு",         "FLASHLIGHT_ON"),
    ("வால்யூம் கூட்டு",            "VOLUME_UP"),
    ("torch on pannu",            "FLASHLIGHT_ON"),
    ("amma ku call pannu",        "CALL_CONTACT"),
    ("wifi on kar do",            "TOGGLE_WIFI"),
    ("what is the weather today", "UNKNOWN"),
    ("kal subah 7 baje alarm",    "SET_ALARM"),
    ("wapas jaao",                "GO_BACK"),
]

passed = 0
failed = 0
results = []

for user_input, expected_intent in tests:
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    try:
        result = subprocess.run(
            [CLI, "-m", GGUF, "--prompt", prompt, "-n", "64",
             "--temp", "0.0", "--log-disable", "-p", prompt],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        # Extract JSON from output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = output[start:end]
            parsed = json.loads(json_str)
            got_intent = parsed.get("intent", "")
            has_params = "params" in parsed
            ok = (got_intent == expected_intent) and has_params
        else:
            got_intent = "PARSE_ERROR"
            ok = False

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results.append((status, user_input[:30], expected_intent, got_intent))

    except Exception as e:
        results.append(("FAIL", user_input[:30], expected_intent, f"ERROR:{e}"))
        failed += 1

print("\n" + "="*70)
print(f"{'STATUS':<6} {'INPUT':<32} {'EXPECTED':<18} {'GOT'}")
print("-"*70)
for status, inp, exp, got in results:
    print(f"{status:<6} {inp:<32} {exp:<18} {got}")
print("="*70)
print(f"\nResult: {passed}/10 PASSED  |  {failed}/10 FAILED")
print("="*70)

if failed > 3:
    print("\n⚠️  WARNING: More than 3 failures. Check training or increase epochs.")
    sys.exit(1)
else:
    print("\n✅ Validation passed!")
PYEOF

python3 validate_gguf.py

# ── Upload final GGUF to S3 ───────────────────────────────────────────────────
echo ""
echo "[Upload] Pushing vanimitra_v1_q4km.gguf to S3..."
aws s3 cp vanimitra_v1_q4km.gguf \
    s3://${BUCKET}/models/gguf/vanimitra_v1_q4km.gguf \
    --region ${REGION}

echo ""
echo "=================================================="
echo "✅ POST-TRAINING PIPELINE COMPLETE"
echo "   Final model: vanimitra_v1_q4km.gguf"
echo "   S3 copy:     s3://${BUCKET}/models/gguf/vanimitra_v1_q4km.gguf"
echo ""
echo "   Share with Dev 2:"
echo "   aws s3 cp s3://${BUCKET}/models/gguf/vanimitra_v1_q4km.gguf ."
echo "   then: adb push vanimitra_v1_q4km.gguf /data/data/com.example.vanimitra/files/"
echo "=================================================="
