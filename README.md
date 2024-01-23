# gaussian-splat
This is a minimal implementation of the rasterization process for Gaussian Splatting. 

Its goal is to present a more digestible version of the original implementation by extracting the logic from the CUDA kernels to basic torch. The rasterization process is done sequentially
by rasterizing one gaussian at a time rather than distributing the rasterization per pixel (and parallelizing it with custom CUDA kernels)/
As a result, it is highly unoptimized and not meant to be used for training/evaluation: rendering with this codebase will take about 5 minutes / image vs. less than a second for the original implementation.



https://github.com/arnaudstiegler/gaussian-splat/assets/26485052/1f08a9c2-f086-40e1-b75e-28317408dc68



# How to run

MipNerf 360 scenes can be found at: `https://jonbarron.info/mipnerf360/`

Trained Gaussian Splatting models can be found at: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`


To try out the rasterization, run the following:
`python rasterize.py --input_dir {MIPNERF_360_PATH} --trained_model_path {GAUSSIAN_MODEL_PATH} --output_path {OUTPUT_PATH} --scene-index {SCENE_INDEX} --scale-factor 2 [--generate_video]`


# FAQ
