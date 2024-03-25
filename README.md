# NetDiffusion: High-Fidelity Synthetic Network Traffic Generation



<img width="1241" alt="Screenshot 2024-02-29 at 3 41 29 PM" src="https://github.com/noise-lab/NetDiffusion_Generator/assets/47127634/804756f9-156e-4796-bea6-00d5d7bb1706">


# NetDiffusion tutorial

This tutorial is based on the [original guide steps](https://github.com/noise-lab/NetDiffusion_Generator/blob/main/README.md) for generating PCAPs with [NetDiffusion](https://dl.acm.org/doi/abs/10.1145/3639037).

## Introduction
NetDiffusion is an innovative tool designed to address the challenges of obtaining high-quality, labeled network traces for machine learning tasks in networking. Privacy concerns, data staleness, and the limited availability of comprehensive datasets have long hindered research and development efforts. NetDiffusion leverages a controlled variant of a Stable Diffusion model to generate synthetic network traffic that not only boasts high fidelity but also adheres to protocol specifications.

Our approach outperforms current state-of-the-art synthetic trace generation methods by producing packet captures that exhibit higher statistical similarity to real network traffic. This improvement is crucial for enhancing machine learning model training and performance, as well as for supporting a wide range of network analysis and testing tasks beyond ML-centric applications.

## Features
* **High-Fidelity Synthetic Data:** Generate network traffic that closely resembles real-world data in terms of statistical properties and protocol compliance.
* **Compatibility:** Produced traces are compatible with common network analysis tools, facilitating easy integration into existing workflows.
* **Versatility:** Supports a wide array of network tasks, extending the utility of synthetic traces beyond machine learning applications.
* **Open Source:** NetDiffusion is open-source, encouraging contributions and modifications from the community to meet diverse needs.

## Installation (Current support for Linux only)

### Access your machine with GPU support and clone the repository
```
# SSH to Linux server via designated port (see following for example)
ssh -L 7860:LocalHost:7860 username@server_address

# Clone the repository
git clone git@github.com:noise-lab/NetDiffusion_Generator.git
```

### Clone the necessary git forks for StableDiffusion support:
```
# Navigate to the project directory
cd NetDiffusion_Generator

# Access the `fine_tune` folder
cd fine_tune

# Remove the empty `kohya_ss_fork @ 8a39d4d` and clone it as follows
git clone https://github.com/Chasexj/kohya_ss_fork.git

# Clone `sd-webui-fork @ 2533cf8` (also inside `fine_tune` folder)
git clone https://github.com/Chasexj/stable-diffusion-webui-fork.git
```

## Set Python virtual environemnt provided by `kohya_ss_fork`
From now on, I recommend to use the Python 3.10.13 virtual environment in `kohya_ss_fork` folder:

```
cd <NetDiffusion main folder>/fine_tune/kohya_ss_fork/
source venv/bin/activate
```


## Import data
Store raw pcaps used for fine-tuning into `NetDiffusion_Generator/data/fine_tune_pcaps` with the application/service labels as the filenames, e.g.,`netflix_01.pcap`.


```
# Navigate to preprocessing dir
cd data_preprocessing/

# Run preprocessing conversions
python3 pcap_to_img.py

# Navigate to fine-tune dir and the khoya subdir for task creation
# (replace the number in 20_network with the average number of pcaps per traffic type used for fine-tuning)
cd ../fine_tune/kohya_ss_fork/model_training/
mkdir -p example_task/{image/20_network,log,model}

# Leverage Stable Diffusion WebUI for initial caption creation
cd ../../sd-webui-fork/stable-diffusion-webui/
# Lunch WebUI
bash webui.sh
```

1. Open the WebUI via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Under `Extras/Batch From Directory`, enter the absolute path for `/NetDiffusion_Generator/data/preprocessed_fine_tune_imgs` and `/NetDiffusion_Generator/fine_tune/kohya_ss_fork/model_training/test_task/image/20_network` as the input/output directories.
2. Under `Extras/batch_from_directory`, set the `scale to` parameter to `width = 816` and `height = 768` for resource-friendly fine-tuning (adjust based on resource availability).
3. Enable the `caption` parameter under `extras/batch_from_directory` and click `generate`.
4. Terminate `webui.sh`


## Install the Python virtual environment's requirements

Run the `setup.sh` for **fine-tuning** (this file is inside [kohya_ss_fork @ 8a39d4d](https://github.com/Chasexj/kohya_ss_fork/tree/8a39d4dc1b7410d38a1373e490b2261af6a6ac27)). The setup will install all the requirements inside a Python 3.10.13 environment:

```
# Navigate to fine-tuning directory
cd kohya_ss_fork
# Grant execution access
chmod +x ./setup.sh
# Set up configuration
./setup.sh
# Set up accelerate environment (gpu and fp16 recommended)
accelerate config
```

>In to the official repository, `pip install -r requirements` step won't work.
    
## Make sure you have the correct CUDA set to `LD_LIBRARY_PATH`:

1. Verify which cuda version utilized by PyTorch in the virtual environment:
```
(venv) (base) thiago@ifsuldeminas-Z390-M-GAMING:~/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork$ python
Python 3.10.13 (main, Aug 25 2023, 13:20:03) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.version.cuda
'11.8'
>>> exit()
(venv) (base) thiago@ifsuldeminas-Z390-M-GAMING:~/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork$
``` 
>As you can see, I need CUDA 11.8.

2.  Find where is `libducadart.so` (if any):
```
(venv) (base) thiago@ifsuldeminas-Z390-M-GAMING:~/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork$ find / -name libcudart.so 2>/dev/null
/home/caio/miniconda3/envs/tf/lib/libcudart.so
/home/caio/miniconda3/pkgs/cuda-cudart-dev-12.2.53-0/lib/libcudart.so
/home/caio/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/libcudart.so
/home/thiago/anaconda3/envs/ydata/lib/libcudart.so
/home/thiago/anaconda3/pkgs/cuda-cudart-dev-12.1.105-0/lib/libcudart.so
/home/thiago/anaconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/libcudart.so
/home/thiago/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/libcudart.so
/home/caue/anaconda3/envs/myenv/lib/libcudart.so
/home/caue/anaconda3/pkgs/cuda-cudart-dev-12.4.99-0/lib/libcudart.so
(venv) (base) thiago@ifsuldeminas-Z390-M-GAMING:~/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork$ 
```
3. Set the correct one. In my case, it should be set as follows:
`export LD_LIBRARY_PATH=/home/thiago/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib`


4. Now, check again if bitsandbytes is working already:
```
(venv) (base) thiago@ifsuldeminas-Z390-M-GAMING:~/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork$ python -m bitsandbytes
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++ BUG REPORT INFORMATION ++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

+++++++++++++++++++ ANACONDA CUDA PATHS ++++++++++++++++++++
/home/thiago/anaconda3/lib/libicudata.so
/home/thiago/anaconda3/envs/ydata/lib/python3.9/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/thiago/anaconda3/envs/ydata/lib/python3.9/site-packages/torch/lib/libtorch_cuda.so
/home/thiago/anaconda3/envs/ydata/lib/python3.9/site-packages/torch/lib/libc10_cuda.so
/home/thiago/anaconda3/envs/ydata/lib/libcudart.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.12_cuda11.8_cudnn8.7.0_0/lib/python3.12/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.12_cuda11.8_cudnn8.7.0_0/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.12_cuda11.8_cudnn8.7.0_0/lib/python3.12/site-packages/torch/lib/libc10_cuda.so
/home/thiago/anaconda3/pkgs/cuda-cudart-dev-12.1.105-0/lib/libcudart.so
/home/thiago/anaconda3/pkgs/icu-58.2-he6710b0_3/lib/libicudata.so
/home/thiago/anaconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib/libcudart.so
/home/thiago/anaconda3/pkgs/icu-73.1-h6a678d5_0/lib/libicudata.so
/home/thiago/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/libcudart.so
/home/thiago/anaconda3/pkgs/llvm-openmp-18.1.1-h4dfa4b3_0/lib/libomptarget.rtl.cuda.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.10_cuda11.8_cudnn8.7.0_0/lib/python3.10/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.10_cuda11.8_cudnn8.7.0_0/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so
/home/thiago/anaconda3/pkgs/pytorch-2.2.1-py3.10_cuda11.8_cudnn8.7.0_0/lib/python3.10/site-packages/torch/lib/libc10_cuda.so

++++++++++++++++++ /usr/local CUDA PATHS +++++++++++++++++++
/usr/local/lib/python3.8/dist-packages/torch/lib/libc10_cuda.so
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda_linalg.so
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cuda.so

+++++++++++++++ WORKING DIRECTORY CUDA PATHS +++++++++++++++
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda_linalg.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/torch/lib/libc10_cuda.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda114_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda122.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda111.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda111_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda121.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda110.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda120_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda110_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda118.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda122_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda115.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda115_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda121_nocublaslt.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda120.so
/home/thiago/git_ariel/NetDiffusion_Generator/fine_tune/kohya_ss_fork/venv/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda114.so

++++++++++++++++++ LD_LIBRARY CUDA PATHS +++++++++++++++++++
 /home/thiago/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib CUDA PATHS 
/home/thiago/anaconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib/libcudart.so

++++++++++++++++++++++++++ OTHER +++++++++++++++++++++++++++
COMPILED_WITH_CUDA = True
COMPUTE_CAPABILITIES_PER_GPU = ['7.5']
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++ DEBUG INFO END ++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Running a quick check that:
    + library is importable
    + CUDA function is callable


WARNING: Please be sure to sanitize sensible info from any such env vars!

SUCCESS!
Installation was successful!
```


## Change the caption prompt for explicit prompt-to-image correlation
The steps below will create a corresponding text file (e.g., `netflix_01.txt`) for the preprocessed data (e.g., `netflix_01.png`).

```
# For example, 'pixelated network data, type-0' refers to NetFlix pcap,
# Adjust the script based on fine-tuning task.
cd NetDiffusion_Generator/fine_tune && python3 caption_changing.py test_task/image/20_network
```

## Fine-Tuning
If you already run the `setup.sh` file inside `kohya_ss_fork` folder, check the installed requirements:

```
# Navigate to fine-tuning directory
cd kohya_ss_fork
# Set up accelerate environment (gpu and fp16 recommended)
accelerate config
# Check the installed requirements with pip
pip list
# Fine-tune interface initialization
bash gui.sh
```

1. Open the fine-tuning interface via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Under `LoRA -> Training`, load the configuration file via the absolute path for `/NetDiffusion_Generator/fine_tune/LoraLowVRAMSettings.json`
3. Under `LoRA\Training\Folders`, enter the absolute paths for `/NetDiffusion_Generator/fine_tune/kohya_ss_fork/model_training/test_task/image`,`/NetDiffusion_Generator/fine_tune/kohya_ss_fork/model_training/test_task/model`, and `/NetDiffusion_Generator/fine_tune/kohya_ss_fork/model_training/test_task/log` for the *Image/Output/Logging* folders, respectively, and adjust the model name if needed.
4. Under `LoRA\Training\Parameters\Basic`, adjust the Max Resolution to match the resolution from data preprocessing, e.g., 816,768 (i.e., width, height).
5. Click on Start Training to begin the fine-tuning. Adjust the fine-tuning parameters as needed due to different generation tasks may have different parameter requirement to yield better synthetic data quality.
6. After the `model saved` message (check the terminal), close the server (Ctrl + C).

> By default, the `LoraLowVRAMSettings.json` file, contains Windows-based file paths for *Image/Output/Logging* files. You can manually configure the correct absolute paths and the Max Resolution in this file and load it again.

## Generation
In the previous steps we generated a file in the model folder provided (i.e., `fine-tune/kohya_ss_fork/model_training/test_task/model`). The filename (by default) is `Addams.safetensors`. If not, replace `Addams` in the first step in the code block below by the correct name:

```
# Copy the fine-tuned LoRA model (adjust path namings as needed) to Stable Diffusion WebUI
cp model_training/test_task/model/Addams.safetensors ../sd-webui-fork/stable-diffusion-webui/models/Lora/
# Navigate to the generation directory
cd ../sd-webui-fork/stable-diffusion-webui/
# Initialize Stable Diffusion WebUI
bash webui.sh
```

1. Open the WebUI via the ssh port on the preferred browser, example address: http://localhost:7860/
2. Install ControlNet extension for the WebUI and restart the WebUI: https://github.com/Mikubill/sd-webui-controlnet. Go to `Extensions/` and place this URL in the according field to install it.
3. To generate an image representation of a network trace, enter the corresponding caption prompt with the LoRA model extension under 'txt2img'. For example 'pixelated network data, type-0 \<lora:Addams:1\>' for NetFlix data.
> Note: the safetensors file rule aforementioned is also applied here \<lora:Addams:1\>
5. Adjust the generation resolution to match the resolution from data preprocessing, e.g., 816,768.
6. Adjust the seed to match the seed used in fine-tuning, default is 1234.
7. Enable Hires.fix to scale to 1088, 1024.
8. From training data, sample a real pcap image (that belongs to the same category as the desired synthetic traffic) as input to the ControlNet interface, and set the Control Type (we recommend canny).
> Download the models from ControlNet 1.1: https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main
>
> You need to download model files ending with ".pth" .
> 
> Download canny model: `wget https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth`
> Put models in your `stable-diffusion-webui\extensions\sd-webui-controlnet\models`. You only need to download "pth" files.
> Do not right-click the filenames in HuggingFace website to download. Some users right-clicked those HuggingFace HTML websites and saved those HTML pages as PTH/YAML files. They are not downloading correct files. Instead, please click the small download arrow “↓” icon in HuggingFace to download.


> **Note: Besides placing/dropping a fine-tune image do not forget to Enable ControlNet before generating the image.**

10. Click Generate to complete the generation. Note that extensive adjustments on the generation and ControlNet parameters may be needed to yield the best generation result as the generation tasks and training data differ from each other.

## Post-Generation Heuristic Correction
Once enough instances of image representations of desired synthetic traffic are generated, place all of such instances under `/NetDiffusion_Generator/data/generated_imgs`.

Navigate to the `/NetDiffusion_Generator/post-generation/` folder and:
```
# Run the following to do the post-generation heuristic for reconversion back to pcaps and protocol compliance checking.
# Adjust the color processing threshold in color_processor.py as required for best generation results.
python3 color_processor.py && python3 img_to_nprint.py && python3 mass_reconstruction.py
```

This completes the post-generation pipeline with the final nprints and pcaps stored in `/NetDiffusion_Generator/data/replayable_generated_nprints` and `/NetDiffusion_Generator/data/replayable_generated_pcaps`, respectively.

## Testing the generated PCAPs
To test whether the generated packets are valid, you can try resending them through your local interface:

```
# Install tcpreplay
sudo apt update
sudo apt install tcpreplay
# Grab the local interface's name
ip a
# Navidate to the generated PCAPs folder
cd NetDiffusion_Generator/data/replayable_generated_pcaps
# Run tcpreplay with a generated PCAP file
sudo tcpreplay --loop=0 --verbose -i eno1 00000-1234.pcap
```

> **Note: Replace `0000-1234.pcap` with the correct filename if necessary. If there is a file named `netflix_5.pcap`, consider it was already there and generated by the original authors. You did not generate this one!**


## Citing NetDiffusion
```
@article{jiang2024netdiffusion,
  title={NetDiffusion: Network Data Augmentation Through Protocol-Constrained Traffic Generation},
  author={Jiang, Xi and Liu, Shinan and Gember-Jacobson, Aaron and Bhagoji, Arjun Nitin and Schmitt, Paul and Bronzino, Francesco and Feamster, Nick},
  journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
  volume={8},
  number={1},
  pages={1--32},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```


