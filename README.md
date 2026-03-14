Training VaaniMitra
===================

Overview
--------
This project focuses on training the Qwen2 0.5B LLM for multilingual compatibility, specifically designed for a custom accessibility application called "VaaniMitra".
Project Structure
The repository contains the following key files:
Setup & Execution

RUN_ME.sh - Main entry point script to execute the training pipeline
00_clean.sh - Cleanup script to prepare the environment

Data & Training Pipeline
------------------------

01_dataset_gen.py - Dataset generation script for preparing training data
02_sagemaker_launch.py - AWS SageMaker launch script for cloud-based training
train.py - Core training script for the Qwen2 model
03_post_training.sh - Post-training processing and model finalization

Technology Stack
-----------------

Primary Language: Python (79.3%)
Shell Scripts: (20.7%)
Base Model: Qwen2 0.5B LLM
Cloud Platform: AWS SageMaker

Purpose
-------
VaaniMitra is an accessibility application that requires multilingual support. This training pipeline fine-tunes the lightweight Qwen2 0.5B model to handle multiple languages effectively while maintaining a small footprint suitable for accessibility use cases.
