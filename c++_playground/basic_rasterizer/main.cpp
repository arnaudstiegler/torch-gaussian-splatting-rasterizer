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

using namespace torch::indexing;

torch::Tensor triangle = torch::tensor({{0.0,0.0,0.0}, {10.0,0.0,0.0}, {0.0,5.0,0.0}});
torch::Tensor triangle_colors = torch::tensor({{255.0,0.0,0.0}, {0.0,255.0,0.0}, {0.0,0.0,255.0}});
torch::Tensor camera_to_world = torch::tensor({
        {1.0, 0.8, 0.0, 0.0},
        {0.2, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.9, 0.0},
        {0.0, 0.0, -1.0, 1.0}
    });
torch::Tensor world_to_camera = torch::inverse(camera_to_world);

int canvas_distance = 1;
int screen_width = 25;
int screen_height = 25;
float pixel_width = 0.01;
float pixel_height = 0.01;


torch::Tensor project_to_camera(torch::Tensor triangle, torch::Tensor world_to_camera){
  torch::Tensor rotation = torch::matmul(world_to_camera.slice(0,0,3).slice(1,0,3), triangle);
  torch::Tensor translation = world_to_camera.index({-1, Slice(0, 3, 1)});
  std::cout << translation << std::endl;
  return  rotation + translation;
}


int main() {
  torch::Tensor projection = project_to_camera(triangle, world_to_camera);

  /*
  projection:
  -9.5238   0.0000   1.1111
  11.9048   0.0000   1.1111
  0.0000   5.5556   1.1111
  */

  std::cout << projection << std::endl;
  return 0;
}