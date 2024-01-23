# gaussian-splat
Reproducing Gaussian Splatting


# How to run

Data can be found at: `https://jonbarron.info/mipnerf360/`

Trained models can be found at: `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip`


To try out the rasterization, run the following:
`python rasterize.py --input_dir data/bonsai/ --trained_model_path data/trained_model/bonsai --output_path data/rendered_images/ --scene-index 2 --scale-factor 2 --generate_video`


# FAQ
