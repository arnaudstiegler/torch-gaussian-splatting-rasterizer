/*
What we need:

World-to-Camera matrix
Triangle(s) coordinates in world referential
Triangle vertices attributes (color only here)
Canvas size

Steps:
- Convert triangle coordinates from world to camera referential
- Project divide onto the screen (get screen coordinates)
- Filter out points that are outside of the canvas
- Convert from Screen Space to NDC space
- Convert from NDC to Raster Space
- For each pixel, find out whether the triangle overlaps (using the edge function) and interpolate attributes using barycentric coordinates (with correction due to non-linearity of the project divide)
*/

#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}