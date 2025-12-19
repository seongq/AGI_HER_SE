import subprocess
import os

dataset_key = ["WSJ0+Chime3_low", "WSJ0+Reverb","VoiceBank-DEMAND" ]

# Define the list of checkpoints, datasets, and N values
datasets = {"WSJ0+Chime3_low":"/data/dataset/WSJ0-CHiME3_low_snr",
            "WSJ0+Reverb":"/data/dataset/WSJ0-CHiME3_derev",
            "VoiceBank-DEMAND": "/data/dataset/VCTK_corpus"}
base_command = "CUDA_VISIBLE_DEVICES=2 python enhancement.py"

N_values = [1, 2, 3,4,5,6,7,8,9,10,50]  # Define the list of N values
pretrained_path = "/data/baseline_interspeech2025/storm_pretrained/pretrained_model_by_author"
models_path = os.listdir(pretrained_path)
for model in models_path:
    if model =="ncsnpp":
        mode = "denoiser-only"
        
    elif model == "sgmse":
        mode = "score-only"
    elif model == "storm":
        mode = "storm"
    model_dataset_path = os.path.join(pretrained_path, model)
    model_dataset_list = os.listdir(model_dataset_path)
    for dataset_name in model_dataset_list:
        if dataset_name in dataset_key:    
            try:
                ckpt_name = os.listdir(os.path.join(pretrained_path, model,dataset_name))
                assert len(ckpt_name) == 1
                ckpt_name = ckpt_name[0]
                ckpt_path = os.path.join(pretrained_path, model, dataset_name, ckpt_name)
                
            except FileNotFoundError:
                pass
            for n in N_values:
                enhanced_dir = f"{dataset_name}_{model}_N_{n}"
                command = (
                    f"{base_command} "
                    f"--test_dir {datasets[dataset_name]} "
                    f"--enhanced_dir {enhanced_dir} "
                    f"--ckpt {ckpt_path} "
                    f"--mode {mode} "
                    f"--N {n}"
                )
                
                if model == "ncsnpp":
                    if n>1:
                        continue
                
                print(f"Running command: {command}")
                
                # Execute the command
                result = subprocess.run(command, shell=True, text=True)
                
                # Check for errors
                if result.returncode != 0:
                    print(f"Error occurred while processing {dataset} with {checkpoint} and N={N}.")
                    print(result.stderr)




