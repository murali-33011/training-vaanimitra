#!/usr/bin/env python3
# 02_sagemaker_launch.py
# Run this from your ml.t3.medium terminal AFTER running 01_dataset_gen.py
# Usage: python 02_sagemaker_launch.py --hf-token hf_YOURTOKEN

import argparse
import boto3
import sagemaker
import os
import time
import json
from sagemaker.pytorch import PyTorch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", required=True, help="Your Hugging Face token (hf_...)")
parser.add_argument("--region", default="us-east-1")
args = parser.parse_args()

REGION = args.region
HF_TOKEN = args.hf_token
BUCKET = "vaanimitra"
ROLE_ARN = "arn:aws:iam::810500877577:role/AmazonSageMaker-ExecutionRole-20260308T154718"

print(f"Region:  {REGION}")
print(f"Bucket:  s3://{BUCKET}")
print(f"Role:    {ROLE_ARN}")

# ── Session ───────────────────────────────────────────────────────────────────
boto_session = boto3.Session(region_name=REGION)
sm_session = sagemaker.Session(boto_session=boto_session, default_bucket=BUCKET)

# ── Step 1: Upload dataset ────────────────────────────────────────────────────
TRAIN_LOCAL = "vanimitra_train.jsonl"
TRAIN_S3_KEY = "datasets/training/vanimitra_train.jsonl"

if not os.path.exists(TRAIN_LOCAL):
    print(f"ERROR: {TRAIN_LOCAL} not found. Run 01_dataset_gen.py first.")
    exit(1)

print(f"\n[1/3] Uploading dataset to s3://{BUCKET}/{TRAIN_S3_KEY} ...")
s3 = boto_session.client("s3")
s3.upload_file(TRAIN_LOCAL, BUCKET, TRAIN_S3_KEY)
print("      Done.")

TRAIN_S3_URI = f"s3://{BUCKET}/datasets/training"

# ── Step 2: Upload train.py ───────────────────────────────────────────────────
print("\n[2/3] Uploading train.py ...")
s3.upload_file("train.py", BUCKET, "scripts/train.py")
print("      Done.")

# ── Step 3: Launch training job ───────────────────────────────────────────────
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
JOB_NAME = f"vanimitra-qwen-{TIMESTAMP}"

print(f"\n[3/3] Launching SageMaker training job: {JOB_NAME}")
print(f"      Instance: ml.g4dn.xlarge")
print(f"      Estimated time: ~45-60 minutes")
print(f"      Estimated cost: ~$0.50")

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",               # uploads current directory (train.py must be here)
    role=ROLE_ARN,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    sagemaker_session=sm_session,
    output_path=f"s3://{BUCKET}/models/lora-adapters/vanimitra-v1",
    checkpoint_s3_uri=f"s3://{BUCKET}/checkpoints/training-runs/{JOB_NAME}",
    hyperparameters={},
    environment={
        "HF_TOKEN": HF_TOKEN,
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
        "HF_HOME": "/tmp/hf_home",
    },
    max_run=7200,  # 2 hour timeout
    keep_alive_period_in_seconds=0,
)

estimator.fit(
    inputs={"train": TRAIN_S3_URI},
    job_name=JOB_NAME,
    wait=False,  # non-blocking — we'll poll below
)

print(f"\n✅ Job submitted: {JOB_NAME}")
print(f"\nMonitoring (Ctrl+C to stop watching, job continues in cloud)...")
print(f"Console: https://console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{JOB_NAME}\n")

# ── Poll for completion ───────────────────────────────────────────────────────
sm_client = boto_session.client("sagemaker")
last_status = None

while True:
    response = sm_client.describe_training_job(TrainingJobName=JOB_NAME)
    status = response["TrainingJobStatus"]
    secondary = response.get("SecondaryStatus", "")

    if status != last_status:
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} | {secondary}")
        last_status = status

    if status in ("Completed", "Failed", "Stopped"):
        break

    time.sleep(30)

if status == "Completed":
    output_path = response["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Output: {output_path}")
    print(f"   Job:    {JOB_NAME}")
    print(f"{'='*60}")
    print(f"\nNext step: bash 03_post_training.sh {JOB_NAME}")

    # Save job info for next script
    with open("job_info.json", "w") as f:
        json.dump({
            "job_name": JOB_NAME,
            "output_path": output_path,
            "bucket": BUCKET,
            "region": REGION,
        }, f, indent=2)
    print("   job_info.json saved for post-training script.")

else:
    failure = response.get("FailureReason", "unknown")
    print(f"\n{'='*60}")
    print(f"❌ TRAINING FAILED")
    print(f"   Status:  {status}")
    print(f"   Reason:  {failure}")
    print(f"   Logs: https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs/log-events/{JOB_NAME}")
    print(f"{'='*60}")
    exit(1)
