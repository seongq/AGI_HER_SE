import numpy as np
import glob
from soundfile import read, write
import pandas as pd
from tensorboard import summary
from tqdm import tqdm
from torchaudio import load, save
import torch
import os
from argparse import ArgumentParser
import time

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.odes import ODERegistry
from sgmse.model import StochasticRegenerationModel, ScoreModel, DiscriminativeModel

from sgmse.util.other import *

import matplotlib.pyplot as plt
from os.path import join
from utils import energy_ratios, ensure_dir, print_mean_std
from pesq import pesq
from pystoi import stoi
EPS_LOG = 1e-10
sr=16000
# Tags
base_parser = ArgumentParser(add_help=False)
parser = ArgumentParser()
for parser_ in (base_parser, parser):
	parser_.add_argument("--test_dir", type=str, required=True, help="Directory containing your corrupted files to enhance.")
	parser_.add_argument("--enhanced_dir", type=str, required=True, help="Where to write your cleaned files.")
	parser_.add_argument("--ckpt", type=str, required=True)
	parser_.add_argument("--mode", required=True, choices=["score-only", "denoiser-only", "storm"])


	parser_.add_argument("--N", type=int, default=5, help="Number of reverse steps")

args = parser.parse_args()

# os.makedirs(args.enhanced_dir, exist_ok=True)

#Checkpoint
checkpoint_file = args.ckpt

# Settings
model_sr = 16000

# Load score model 
if args.mode == "storm":
	model_cls = StochasticRegenerationModel
elif args.mode == "score-only":
	model_cls = ScoreModel
elif args.mode == "denoiser-only":
	model_cls = DiscriminativeModel

model = model_cls.load_from_checkpoint(
	checkpoint_file, base_dir="",
	batch_size=1, num_workers=0, kwargs=dict(gpu=False)
)
model.eval(no_ema=False)
model.cuda()
# num_parameters = count_parameters(model)
# print(f"Total number of trainable parameters: {num_parameters}")


clean_dir = join(args.test_dir, "test", "clean")
noisy_dir = join(args.test_dir, "test", "noisy")
target_dir = f"/data/{args.enhanced_dir}/"
ensure_dir(target_dir + "files/")

noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
# print(noisy_files)

data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
for cnt, noisy_file in tqdm.tqdm(enumerate(noisy_files)):
	filename = noisy_file.split('/')[-1]
	
	# Load wav
	x, _ = load(join(clean_dir, filename))
	y, _ = load(noisy_file)
	

	x_hat = model.enhance(y,  N=args.N)

	x_hat = x_hat.squeeze().cpu().numpy()


	x = x.squeeze().cpu().numpy()
	y = y.squeeze().cpu().numpy()
	n = y - x
	# Write enhanced wav file
	write(target_dir + "files/" + filename, x_hat, 16000)

	# Append metrics to data frame
	data["filename"].append(filename)
	try:
		p = pesq(sr, x, x_hat, 'wb')
	except: 
		p = float("nan")
	data["pesq"].append(p)
	data["estoi"].append(stoi(x, x_hat, sr, extended=True))
	data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
	data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
	data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

# Save results as DataFrame
df = pd.DataFrame(data)
df.to_csv(join(target_dir, "_results.csv"), index=False)

# Save average results
text_file = join(target_dir, "_avg_results.txt")
with open(text_file, 'w') as file:
	file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
	file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
	file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
	file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
	file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

# Save settings
text_file = join(target_dir, "_settings.txt")
with open(text_file, 'w') as file:
	file.write("checkpoint file: {}\n".format(args.ckpt))
	
	
	file.write("N: {}\n".format(args.N))
	

	

	
	