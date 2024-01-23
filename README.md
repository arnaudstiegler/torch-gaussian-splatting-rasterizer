# gaussian-splat
This is a minimal implementation of the rasterization process for Gaussian Splatting. 

Its goal is to present a more digestible version of the original implementation by extracting the logic from the CUDA kernels to basic torch. The rasterization process is done sequentially
by rasterizing one gaussian at a time rather than distributing the rasterization per pixel (and parallelizing it with custom CUDA kernels)/
As a result, it is highly unoptimized and not meant to be used for training/evaluation: rendering with this codebase will take about 5 minutes / image vs. less than a second for the original implementation.

<video controls="" width="800" height="500" muted="" loop="" autoplay="">
<source src="https://github.com/arnaudstiegler/gaussian-splat/blob/main/assets/render_video.mp4" type="video/mp4">
</video>

# How to run

MipNerf 360 scenes can be found at: `https://jonbarron.info/mipnerf360/`

Trained Gaussian Splatting models can be found at: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`


To try out the rasterization, run the following:
`python rasterize.py --input_dir {MIPNERF_360_PATH} --trained_model_path {GAUSSIAN_MODEL_PATH} --output_path {OUTPUT_PATH} --scene-index {SCENE_INDEX} --scale-factor 2 [--generate_video]`


# FAQ
