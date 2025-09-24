#!/usr/bin/env python3
# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Domain finetune round-robin scheduler for GPU 0 and GPU 1
This script uses multiprocessing to run domain finetune commands in parallel,
distributing them across 2 GPUs with a process pool limited to 2 workers.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Configuration
CONFIGS = [
    "adme_microsom_stab_h",
    "adme_microsom_stab_r",
    "adme_permeability",
    "adme_ppb_h",
    "adme_ppb_r",
    "adme_solubility",
    "astrazeneca_CL",
    "astrazeneca_LogD74",
    "astrazeneca_PPB",
    "astrazeneca_Solubility",
]
ENV_VARS = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
    "WANDB_LOG_MODEL": "end",
    "WANDB_MODE": "online",
    "HF_HUB_OFFLINE": "0",
    "WANDB_PROJECT": "ModChemBERT-Domain-Finetune",
}

BASE_CMD = [".venv/bin/python", "scripts/train_modchembert.py", "--config-dir=conf/task-adaptation-ft-datasets"]


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    logger = logging.getLogger("task_finetune_scheduler")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_dir / "scheduler.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def run_training_job(
    config_name: str,
    gpu_id: int,
    log_dir: Path,
    cuda_enabled: bool,
    pretrained_model_path: str | None = None,
    hyperopt_enabled: bool = False,
) -> tuple[str, int, bool, str]:
    """
    Run a single training job on a specific GPU or CPU

    Args:
        config_name: Name of the config to use for training
        gpu_id: GPU ID to use for training, or -1 for CPU
        log_dir: Directory to save logs
        cuda_enabled: Whether CUDA is enabled
        pretrained_model_path: Optional path to pretrained model
        hyperopt_enabled: Whether to enable hyperparameter optimization

    Returns:
        Tuple of (config_name, gpu_id, success, log_file_path)
    """
    # Setup environment
    env = os.environ.copy()
    env.update(ENV_VARS)

    if cuda_enabled and gpu_id >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    # Setup logging
    device_str = f"GPU {gpu_id}" if cuda_enabled and gpu_id >= 0 else "CPU"
    log_file = log_dir / f"{device_str.lower().replace(' ', '')}_{config_name}.log"

    # Build command
    cmd = BASE_CMD + [f"--config-name={config_name}"]

    # Add pretrained model override if provided
    if pretrained_model_path:
        cmd.append(f"modchembert.pretrained_model={pretrained_model_path}")

    # Add hyperopt override if enabled
    if hyperopt_enabled:
        cmd.append("hyperopt.enabled=true")

    try:
        with open(log_file, "w") as f:
            f.write(f"Starting training for {config_name} on {device_str}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Environment: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            f.write("=" * 50 + "\n\n")
            f.flush()

            # Run the training command
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd())

            # Wait for completion
            return_code = process.wait()

            f.write(f"\n\nTraining completed with return code: {return_code}\n")

        success = return_code == 0
        return (config_name, gpu_id, success, str(log_file))

    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"\n\nError occurred: {str(e)}\n")
        return (config_name, gpu_id, False, str(log_file))


def worker_function(args: tuple[str, int, Path, bool, str | None, bool]) -> tuple[str, int, bool, str]:
    """Worker function for multiprocessing"""
    config_name, gpu_id, log_dir, cuda_enabled, pretrained_model_path, hyperopt_enabled = args
    return run_training_job(config_name, gpu_id, log_dir, cuda_enabled, pretrained_model_path, hyperopt_enabled)


def analyze_log_file(log_file_path: str) -> str:
    """Analyze log file to determine job status"""
    try:
        if not os.path.exists(log_file_path):
            return "LOG_NOT_FOUND"

        with open(log_file_path) as f:
            content = f.read().lower()

        # Check for success indicators
        success_indicators = [
            "training completed with return code: 0",
            "training complete",
            "model saved successfully",
            "training finished",
        ]

        # Check for error indicators
        error_indicators = ["error", "exception", "failed", "traceback", "return code: 1", "return code: 2"]

        if any(indicator in content for indicator in success_indicators):
            return "SUCCESS"
        elif any(indicator in content for indicator in error_indicators):
            return "FAILED"
        else:
            return "UNKNOWN"

    except Exception:
        return "ERROR_READING_LOG"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Domain finetune round-robin scheduler", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num-workers", type=int, help="Number of parallel workers to use")

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Enable CUDA (use GPUs). If not specified, CUDA_VISIBLE_DEVICES will be set to -1",
    )

    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to pretrained model. If provided, will be passed as modchembert.pretrained_model override",
    )

    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Enable hyperparameter optimization by adding hyperopt.enabled=true to the command",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"outputs/task_finetune_{timestamp}")
    logger = setup_logging(log_dir)

    logger.info("Starting domain finetune round-robin scheduler")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Total configs to process: {len(CONFIGS)}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"CUDA enabled: {args.cuda}")
    if args.cuda:
        logger.info("Using GPUs: [0, 1, ...]")
    else:
        logger.info("CUDA disabled - running on CPU")
    if args.pretrained_model_path:
        logger.info(f"Using pretrained model: {args.pretrained_model_path}")
    logger.info(f"Hyperparameter optimization enabled: {args.hyperopt}")
    logger.info("-" * 50)

    # Create task arguments with round-robin GPU assignment
    tasks = []
    for i, config in enumerate(CONFIGS):
        gpu_id = i % args.num_workers if args.cuda else -1
        tasks.append((config, gpu_id, log_dir, args.cuda, args.pretrained_model_path, args.hyperopt))

    # Track results
    results = []
    completed_configs = set()

    # Use ProcessPoolExecutor with specified number of workers
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_config = {executor.submit(worker_function, task): task[0] for task in tasks}

        # Process completed tasks
        for future in as_completed(future_to_config):
            config_name = future_to_config[future]

            try:
                config_name, gpu_id, success, log_file = future.result()
                results.append((config_name, gpu_id, success, log_file))
                completed_configs.add(config_name)

                status = "SUCCESS" if success else "FAILED"
                logger.info(f"Completed {config_name} on GPU {gpu_id} - {status}")

            except Exception as e:
                logger.error(f"Error processing {config_name}: {str(e)}")
                results.append((config_name, -1, False, ""))

    # Wait a moment for any final writes
    time.sleep(2)

    # Display summary
    logger.info("All domain finetune jobs completed!")
    logger.info(f"Logs are available in: {log_dir}")
    logger.info("")
    logger.info("=== EXECUTION SUMMARY ===")

    success_count = 0
    failed_count = 0

    for config_name, gpu_id, success, log_file in results:
        if success:
            logger.info(f"✓ {config_name} (GPU {gpu_id}) - SUCCESS")
            success_count += 1
        else:
            # Double-check by analyzing log file
            if log_file and os.path.exists(log_file):
                status = analyze_log_file(log_file)
                if status == "SUCCESS":
                    logger.info(f"✓ {config_name} (GPU {gpu_id}) - SUCCESS")
                    success_count += 1
                else:
                    logger.info(f"✗ {config_name} (GPU {gpu_id}) - FAILED (check {log_file})")
                    failed_count += 1
            else:
                logger.info(f"✗ {config_name} (GPU {gpu_id}) - FAILED (no log file)")
                failed_count += 1

    logger.info("")
    logger.info(f"Summary: {success_count} succeeded, {failed_count} failed out of {len(CONFIGS)} total")

    # Return appropriate exit code
    if failed_count == 0:
        logger.info("All jobs completed successfully!")
        sys.exit(0)
    else:
        logger.warning(f"{failed_count} jobs failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
