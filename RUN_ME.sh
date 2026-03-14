#!/bin/bash
# RUN_ME.sh — Master guide. Read this top to bottom. Do exactly this.
# Every command is copy-paste ready.

cat << 'BANNER'
╔══════════════════════════════════════════════════════════╗
║           VANIMITRA — COMPLETE RUN GUIDE                 ║
║    Read top to bottom. Every command is exact.           ║
╚══════════════════════════════════════════════════════════╝
BANNER

echo "
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 0 — APPLY S3 BUCKET POLICY (do this ONCE in AWS Console)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Go to: https://s3.console.aws.amazon.com/s3/buckets/vaanimitra
2. Click: Permissions tab
3. Click: Bucket policy → Edit
4. Paste the entire contents of s3_bucket_policy.json
5. Click: Save changes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — CLEAN + SETUP (in your ml.t3.medium terminal NOW)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 1: Wipe everything, make clean workspace
bash 00_clean.sh
cd ~/vanimitra

# Step 2: Copy all scripts into this directory
# (Upload all files from Claude to ~/vanimitra using the notebook file browser
#  OR run: cp /path/to/downloaded/*.py ~/vanimitra/ )

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — GENERATE DATASET (~2 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cd ~/vanimitra
python3 01_dataset_gen.py

# Expected output:
#   ✅ Written: 800+ examples
#   ❌ Errors:  0
#   📄 Output:  vanimitra_train.jsonl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3 — LAUNCH TRAINING ON ml.g4dn.xlarge (~1 hour)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Replace hf_YOURTOKEN with your actual token from huggingface.co/settings/tokens
python3 02_sagemaker_launch.py --hf-token hf_YOURTOKEN

# This will:
#   - Upload vanimitra_train.jsonl to s3://vaanimitra/datasets/training/
#   - Launch training job on ml.g4dn.xlarge (~\$0.50)
#   - Monitor and print status every 30 seconds
#   - Save job_info.json when done

# While training runs, you can close terminal — job runs in cloud.
# Check status: https://console.aws.amazon.com/sagemaker/home#/jobs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 4 — POST-TRAINING: MERGE + GGUF + QUANTIZE (~30 min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Run ONLY after Phase 3 completes (status = Completed)
# Make sure job_info.json is in ~/vanimitra/

cd ~/vanimitra
bash 03_post_training.sh

# This will:
#   1. Download LoRA adapters from S3
#   2. Merge LoRA into base Qwen model
#   3. Clone + build llama.cpp
#   4. Convert to GGUF (f16)
#   5. Quantize to Q4_K_M
#   6. Validate on 10 test cases (prints PASS/FAIL)
#   7. Upload vanimitra_v1_q4km.gguf to S3

# Final file: ~/vanimitra/vanimitra_v1_q4km.gguf

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 5 — SHARE WITH DEV 2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Dev 2 can download the model directly from S3:
aws s3 cp s3://vaanimitra/models/gguf/vanimitra_v1_q4km.gguf .

# Then on their machine with the phone connected:
# adb push vanimitra_v1_q4km.gguf /data/data/com.example.vanimitra/files/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If training fails with 'Access Denied on S3':
  → Re-apply s3_bucket_policy.json (Phase 0)

If training fails with 'Repository not found' or '401':
  → Your HF token is wrong or you haven't accepted Qwen license
  → Go to: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct and click Agree

If training fails with 'ResourceLimitExceeded' for ml.g4dn.xlarge:
  → You have quota=1 but something else is running. Check:
    https://console.aws.amazon.com/sagemaker/home#/jobs
  → Wait for it to finish then rerun Phase 3

If 03_post_training.sh fails at llama.cpp build:
  → Run: sudo yum install -y cmake gcc gcc-c++ make   (Amazon Linux)
  → Then re-run bash 03_post_training.sh

If llama-quantize binary not found:
  → ls llama.cpp/build/bin/   to see actual binary name
  → Edit 03_post_training.sh line with llama-quantize to match

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 0 (S3 policy):     2 min
Phase 1 (setup):         3 min
Phase 2 (dataset gen):   2 min
Phase 3 (training):      45-60 min
Phase 4 (post-train):    25-35 min
Phase 5 (share):         2 min
────────────────────────────────
TOTAL:                   ~90 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"
