# automate_train.py

import subprocess

def run_training(model_type, e, b):
    # Base command
    command = f"python train.py --model_type {model_type} --e {e} --b {b}"

    # Use subprocess to execute the command
    print(f"Executing command: {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()  # Wait for the process to complete

if __name__ == '__main__':
    model_type = "alexnet"
    run_training(model_type=model_type, e=30, b=8)
    run_training(model_type=model_type, e=30, b=16)
    run_training(model_type=model_type, e=30, b=32)
    run_training(model_type=model_type, e=30, b=64)
    run_training(model_type=model_type, e=30, b=128)
    run_training(model_type=model_type, e=30, b=256)
    run_training(model_type=model_type, e=30, b=512)

    model_type = "resnet18"
    run_training(model_type=model_type, e=30, b=8)
    run_training(model_type=model_type, e=30, b=16)
    run_training(model_type=model_type, e=30, b=32)
    run_training(model_type=model_type, e=30, b=64)
    run_training(model_type=model_type, e=30, b=128)
    run_training(model_type=model_type, e=30, b=256)
    run_training(model_type=model_type, e=30, b=512)
    
    model_type = "vgg16"
    run_training(model_type=model_type, e=30, b=8)
    run_training(model_type=model_type, e=30, b=16)
    run_training(model_type=model_type, e=30, b=32)
    
