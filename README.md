<h1 align="center">FramePack Studio</h1>


[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/MtuM7gFJ3V)[![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)](https://www.patreon.com/ColinU)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/colinurbs/FramePack-Studio)

FramePack Studio is an AI video generation application based on FramePack that strives to provide everything you need to create high quality video projects. 

![screencapture-127-0-0-1-7860-2025-06-12-19_50_37](https://github.com/user-attachments/assets/b86a8422-f4ce-452b-80eb-2ba91945f2ea)
![screencapture-127-0-0-1-7860-2025-06-12-19_52_33](https://github.com/user-attachments/assets/ebfb31ca-85b7-4354-87c6-aaab6d1c77b1)


## Current Features

- **F1, Original and Video Extension Generations**: Run all in a single queue
- **End Frame Control for 'Original' Model**: Provides greater control over generations
- **Upscaling and Post-processing**
- **Timestamped Prompts**: Define different prompts for specific time segments in your video
- **Prompt Blending**: Define the blending time between timestamped prompts
- **LoRA Support**: Works with most (all?) Hunyuan Video LoRAs
- **Queue System**: Process multiple generation jobs without blocking the interface. Import and export queues.
- **Metadata Saving/Import**: Prompt and seed are encoded into the output PNG, all other generation metadata is saved in a JSON file that can be imported later for similar generations.
- **Custom Presets**: Allow quick switching between named groups of parameters. A custom Startup Preset can also be set.
- **I2V and T2V**: Works with or without an input image to allow for more flexibility when working with standard Hunyuan Video LoRAs
- **Latent Image Options**: When using T2V you can generate based on a black, white, green screen, or pure noise image


## Fresh Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
- 16GB System Memory (32GB+ strongly recommended)
- 80GB+ of storage (including ~25GB for each model family: Original and F1)

### Setup
There are Multiple ways to install FramePack Studio.

- If you want a very easy installation and plan to use other AI tools, Pinokio might be a good choice. It manages installations for a wide range of popular AI projects. The Pinokio Installation sets up everything you need, including *Sage Attention*, which speeds up video generation.
- If you're already comfortable installing python based projects from the terminal, or need to share the heavy dependencies between installations, try Manual Installation.

#### Pinokio Installation

1. First install Pinokio (a download and 2 clicks) https://www.pinokio.co/
   Note, this is the new Pinokio domain, as announced here: ⁠Pinokio⁠

2. With Pinokio installed, open it and in the upper right corner go to Settings. There, you'll set Home to a directory with lots of space for all your pinokio apps. FramePack Studio requires 80GB+.

3. With Pinokio installed, the direct link to install FramePack Studio is:
   https://www.pinokio.co/item.html?uri=https%3A%2F%2Fgithub.com%2Fcolinurbs%2FFP-Studio&parent_frame=&theme=null

   Click One-Click Install with Pinokio
   Click Download

   In Pinokio, click Install

4. FramePack Studio should now be listed on your Pinokio home page, ready to *Start*. The first start will be slow, with a download.

   Note: The first time you [Add To Queue] to generate video with a model family (Original, F1) will be very slow, downloading the ~25GB models before it begins. Expect that and don't cancel thinking it's broken.

#### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/colinurbs/FramePack-Studio.git
   cd FramePack-Studio
   ```

2. Install PyTorch:

   Go to the [PyTorch Getting Started](https://pytorch.org/get-started/locally/) page and install PyTorch according to your system setup.
   For example, if using CUDA 12.6 on Windows:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

3. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

#### Pinokio Users
Open Pinokio and click the FramePack Studio project on your home page. Then *Start* on the side bar.

#### Manual Users:
Run the studio interface:

```bash
python studio.py
```

Additional command line options:
- `--share`: Create a public Gradio link to share your interface
- `--server`: Specify the server address (default: 0.0.0.0)
- `--port`: Specify a custom port
- `--inbrowser`: Automatically open the interface in your browser
- `--offline`: Disable HF model checks to allow use without internet

## LoRAs

Add LoRAs to the /loras/ folder at the root of the installation. Select the LoRAs you wish to load and set the weights for each generation. Most Hunyuan LoRAs were originally trained for T2V, it's often helpful to run a T2V generation to ensure they're working before using input images.

NOTE: slow lora loading is a known issue

## Working with Timestamped Prompts

You can create videos with changing prompts over time using the following syntax:

```
[0s: A serene forest with sunlight filtering through the trees ]
[5s: A deer appears in the clearing ]
[10s: The deer drinks from a small stream ]
```

Each timestamp defines when that prompt should start influencing the generation. The system will (hopefully) smoothly transition between prompts for a cohesive video.

## Credits
Many thanks to [Lvmin Zhang](https://github.com/lllyasviel) for the absolutely amazing work on the original [FramePack](https://github.com/lllyasviel/FramePack) code!

Thanks to [Rickard Edén](https://github.com/neph1) for the LoRA code and their general contributions to this growing FramePack scene!

Thanks to [Zehong Ma](https://github.com/Zehong-Ma) for [MagCache](https://github.com/Zehong-Ma/MagCache): Fast Video Generation with Magnitude-Aware Cache!

Thanks to everyone who has joined the Discord, reported a bug, sumbitted a PR, or helped with testing!



    @article{zhang2025framepack,
        title={Packing Input Frame Contexts in Next-Frame Prediction Models for Video Generation},
        author={Lvmin Zhang and Maneesh Agrawala},
        journal={Arxiv},
        year={2025}
    }

    @misc{zhang2025packinginputframecontext,
        title={Packing Input Frame Context in Next-Frame Prediction Models for Video Generation}, 
        author={Lvmin Zhang and Maneesh Agrawala},
        year={2025},
        eprint={2504.12626},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2504.12626}
    }