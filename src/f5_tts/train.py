#!/usr/bin/env python3
import subprocess
import sys

def main():
    # 使用 bash 执行所有命令
    full_command = """
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate f5-tts
    export WANDB_API_KEY=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
    wandb login
    cd /yangliusha03/panyuanhao/F5-TTS-main
    accelerate launch src/f5_tts/train/train_ip_adapter.py --config-name F5TTS_ip_adapter
    """
    
    # 执行完整的命令序列
    result = subprocess.run(
        ["bash", "-c", full_command],
        text=True
    )
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()