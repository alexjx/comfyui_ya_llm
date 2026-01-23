#!/usr/bin/env python3
"""
Convert Qwen3-VL Hugging Face Model to GGUF Format

Converts a Qwen3-VL safetensors model to GGUF format:
- Q8_0 quantized main model
- F16 (non-quantized) vision projector (mmproj)

Requirements:
- llama.cpp convert_hf_to_gguf.py script
- llama.cpp llama-quantize binary

Usage:
    python convert_qwen3vl_to_gguf.py \
        --convert-script /path/to/llama.cpp/convert_hf_to_gguf.py \
        --quantize-bin /path/to/llama.cpp/llama-quantize \
        --model-dir /nas/xinj/models/llm/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated \
        --output-dir /home/xinj/ComfyUI/models/LLM
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_step(message):
    print(f"\n{'=' * 60}")
    print(f"  {message}")
    print("=" * 60)


def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n→ {description}")

    print(f"  Command: {' '.join(str(c) for c in cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ ERROR: Command not found: {cmd[0]}")
        return False


def get_file_size(path):
    """Get human-readable file size"""
    size = os.path.getsize(path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-VL model to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python convert_qwen3vl_to_gguf.py \\
      --convert-script ~/llama.cpp/convert_hf_to_gguf.py \\
      --quantize-bin ~/llama.cpp/llama-quantize

  # With custom paths
  python convert_qwen3vl_to_gguf.py \\
      --convert-script /path/to/convert_hf_to_gguf.py \\
      --quantize-bin /path/to/llama-quantize \\
      --model-dir /path/to/model \\
      --output-dir /path/to/output \\
      --name MyModel
        """,
    )

    # Required tools
    parser.add_argument(
        "--convert-script",
        required=True,
        help="Path to llama.cpp convert_hf_to_gguf.py script",
    )
    parser.add_argument(
        "--quantize-bin", required=True, help="Path to llama.cpp llama-quantize binary"
    )

    # Model paths
    parser.add_argument(
        "--model-dir",
        default="/nas/xinj/models/llm/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        help="Path to Hugging Face model directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/xinj/ComfyUI/models/LLM",
        help="Output directory for GGUF files (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        default="Huihui-Qwen3-VL-8B-Instruct",
        help="Base name for output files (default: %(default)s)",
    )

    # Options
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep F16 intermediate file (default: delete after quantization)",
    )

    args = parser.parse_args()

    # Validate paths
    convert_script = Path(args.convert_script)
    quantize_bin = Path(args.quantize_bin)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    print_step("Qwen3-VL to GGUF Conversion")
    print("\nConfiguration:")
    print(f"  Model:      {model_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Name:       {args.name}")
    print(f"  Convert:    {convert_script}")
    print(f"  Quantize:   {quantize_bin}")

    # Validate
    errors = []
    if not convert_script.exists():
        errors.append(f"Convert script not found: {convert_script}")
    if not quantize_bin.exists():
        errors.append(f"Quantize binary not found: {quantize_bin}")
    if not model_dir.exists():
        errors.append(f"Model directory not found: {model_dir}")
    if not (model_dir / "config.json").exists():
        errors.append(f"config.json not found in {model_dir}")

    if errors:
        print("\n✗ Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\n✓ All paths validated")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    f16_output = output_dir / f"{args.name}-F16.gguf"
    mmproj_output = output_dir / f"mmproj-{args.name}-F16.gguf"
    q8_output = output_dir / f"{args.name}-Q8_0.gguf"

    # Step 1: Convert to F16
    print_step("Step 1/3: Convert to F16 GGUF")

    if f16_output.exists():
        print(f"✓ F16 file already exists: {f16_output.name}")
        print(f"  Size: {get_file_size(f16_output)}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("  Skipping F16 conversion")
        else:
            f16_output.unlink()
            if not run_command(
                [
                    sys.executable,
                    str(convert_script),
                    str(model_dir),
                    "--outfile",
                    str(f16_output),
                    "--outtype",
                    "f16",
                ],
                "Converting to F16...",
            ):
                return 1
            print(f"✓ Created: {f16_output.name} ({get_file_size(f16_output)})")
    else:
        if not run_command(
            [
                sys.executable,
                str(convert_script),
                str(model_dir),
                "--outfile",
                str(f16_output),
                "--outtype",
                "f16",
            ],
            "Converting to F16...",
        ):
            return 1
        print(f"✓ Created: {f16_output.name} ({get_file_size(f16_output)})")

    # Step 2: Extract mmproj
    print_step("Step 2/3: Extract Vision Projector (mmproj)")

    if mmproj_output.exists():
        print(f"✓ mmproj file already exists: {mmproj_output.name}")
        print(f"  Size: {get_file_size(mmproj_output)}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("  Skipping mmproj extraction")
        else:
            mmproj_output.unlink()
            if not run_command(
                [
                    sys.executable,
                    str(convert_script),
                    str(model_dir),
                    "--outfile",
                    str(mmproj_output),
                    "--outtype",
                    "f16",
                    "--mmproj",
                ],
                "Extracting mmproj...",
            ):
                return 1
            print(f"✓ Created: {mmproj_output.name} ({get_file_size(mmproj_output)})")
    else:
        if not run_command(
            [
                sys.executable,
                str(convert_script),
                str(model_dir),
                "--outfile",
                str(mmproj_output),
                "--outtype",
                "f16",
                "--mmproj",
            ],
            "Extracting mmproj...",
        ):
            return 1
        print(f"✓ Created: {mmproj_output.name} ({get_file_size(mmproj_output)})")

    # Step 3: Quantize to Q8_0
    print_step("Step 3/3: Quantize to Q8_0")

    if q8_output.exists():
        print(f"✓ Q8_0 file already exists: {q8_output.name}")
        print(f"  Size: {get_file_size(q8_output)}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != "y":
            print("  Skipping Q8_0 quantization")
        else:
            q8_output.unlink()
            if not run_command(
                [str(quantize_bin), str(f16_output), str(q8_output), "Q8_0"],
                "Quantizing to Q8_0...",
            ):
                return 1
            print(f"✓ Created: {q8_output.name} ({get_file_size(q8_output)})")
    else:
        if not run_command(
            [str(quantize_bin), str(f16_output), str(q8_output), "Q8_0"],
            "Quantizing to Q8_0...",
        ):
            return 1
        print(f"✓ Created: {q8_output.name} ({get_file_size(q8_output)})")

    # Summary
    print_step("Conversion Complete!")
    print("\nOutput files:")
    if q8_output.exists():
        print(
            f"  ✓ Main Model (Q8_0):        {q8_output.name} ({get_file_size(q8_output)})"
        )
    if mmproj_output.exists():
        print(
            f"  ✓ Vision Projector (F16):   {mmproj_output.name} ({get_file_size(mmproj_output)})"
        )
    if f16_output.exists():
        print(
            f"  • Intermediate (F16):       {f16_output.name} ({get_file_size(f16_output)})"
        )

    # Cleanup
    if f16_output.exists() and not args.keep_f16:
        print(
            f"\nIntermediate F16 file uses {get_file_size(f16_output)} of disk space."
        )
        try:
            response = input("Delete it to save space? (y/n): ").strip().lower()
            if response == "y":
                f16_output.unlink()
                print(f"✓ Deleted {f16_output.name}")
        except (EOFError, OSError):
            # Non-interactive mode (background execution)
            print("Running in non-interactive mode, skipping cleanup prompt.")
            print(f"To delete manually: rm {f16_output}")

    # Usage instructions
    print("\n" + "=" * 60)
    print("Usage in ComfyUI:")
    print("=" * 60)
    print("  1. Restart ComfyUI")
    print(f"  2. Select model: {args.name}-Q8_0.gguf")
    print(f"  3. Select clip_model: mmproj-{args.name}-F16.gguf")
    print("  4. Connect images and prompt")
    print("  5. Set n_gpu_layers=-1 for GPU acceleration")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
