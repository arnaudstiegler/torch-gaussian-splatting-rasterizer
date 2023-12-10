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
#include <opencv2/opencv.hpp>

using namespace torch::indexing;

torch::Tensor project_to_camera(torch::Tensor triangle, torch::Tensor world_to_camera){
  torch::Tensor rotation = torch::matmul(world_to_camera.slice(0,0,3).slice(1,0,3), triangle);
  torch::Tensor translation = world_to_camera.index({-1, Slice(0, 3, 1)});
  return  rotation + translation;
}

torch::Tensor perspective_divide(torch::Tensor triangle, int screen_distance){
  /*
  The formula is P'(x) = (P(x)*Screen_distance) / P(z)
  */
  torch::Tensor x_y_coordinates = triangle.index({Slice(), Slice(None, 2)});
  torch::Tensor z_coordinates = triangle.index({Slice(), Slice(2,3)});
  torch::Tensor screen_projection = torch::divide(x_y_coordinates, z_coordinates) * screen_distance;

  // Add back the z-coordinates as this will be used downstream
  return torch::concat({screen_projection, z_coordinates}, 1);
}

torch::Tensor project_to_NDC(torch::Tensor triangle, int screen_height, int screen_width){
  // In-place multiplication and substraction
  triangle.index({Slice(), Slice(0,1)}).sub_({screen_width / 2.0}).mul_(1.0/screen_width);
  triangle.index({Slice(), Slice(1,2)}).sub_({screen_height / 2.0}).mul_(1.0/screen_height);

  std::cout << 1/screen_width << std::endl;

  return triangle;
}

torch::Tensor project_to_raster_space(torch::Tensor triangle, int screen_height, int screen_width, float pixel_height, float pixel_width){
  /*
  We undo what we did for NDC space (the normalization), might be missing a piece
  */
  triangle.index({Slice(), Slice(0,1)}).add_({screen_width / 2.0}).mul_(screen_width).mul_(1.0/pixel_width);
  triangle.index({Slice(), Slice(1,2)}).add_({screen_height / 2.0}).mul_(screen_height).mul_(1.0/pixel_height);

  return triangle.to(torch::kLong);
}


torch::Tensor generate_image_tensor(torch::Tensor triangle, int screen_height, int screen_width, float pixel_height, float pixel_width){
  long size_x = static_cast<long>(screen_height / pixel_height);
  long size_y = static_cast<long>(screen_width / pixel_width);
  torch::Tensor black_screen = torch::zeros({size_x, size_y});

  // Marking 1 point for now
  black_screen.index({triangle.index({0,0}).item<int>(), triangle.index({0,1}).item<int>()}) = 255;

  std::cout << black_screen.index({triangle.index({0,0}).item<int>(), triangle.index({0,1}).item<int>()});

  return black_screen;
}


int main() {
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


  torch::Tensor projection = project_to_camera(triangle, world_to_camera);

  // TODO: double-check that this is correct
  /*
  projection:
  -9.5238   0.0000   1.1111
  11.9048   0.0000   1.1111
  0.0000   5.5556   1.1111
  */

  torch::Tensor screen_projection = perspective_divide(projection, canvas_distance);

  /*
  Screen projection

  -8.5714   0.0000   1.1111
  10.7143   0.0000   1.1111
  0.0000   5.0000   1.1111
  */

  // At this point, we should be filtering out the vertices that are outside the canvas, but we'll skip for now

  std::cout << screen_projection << std::endl;

  torch::Tensor NDC_projection = project_to_NDC(screen_projection, screen_height, screen_width);

  /*
  NDC_projection (into -1 / 1)
  -0.8429 -0.5000  1.1111
  -0.0714 -0.5000  1.1111
  -0.5000 -0.3000  1.1111
  */

  std::cout << NDC_projection << std::endl;

  // We isolate the z-coordinates before casting to Long for rasterization
  torch::Tensor z_buffer = NDC_projection.index({Slice(), Slice(2, 3)});
  
  torch::Tensor raster_space_projection = project_to_raster_space(NDC_projection, screen_height, screen_width, pixel_height, pixel_width);
  /*
  raster_space_projection

  29142  30000      1
  31071  30000      1
  30000  30500      1
  */
  
  std::cout << raster_space_projection << std::endl;

  torch::Tensor screen = generate_image_tensor(triangle, screen_height, screen_width, pixel_height, pixel_width);

  screen = screen.to(torch::kU8);
  // Ensure tensor is contiguous
  screen = screen.contiguous();

  // Convert the tensor to a Mat
  cv::Mat image(screen.sizes()[0], screen.sizes()[1], CV_8UC1, screen.data_ptr<uchar>());

  // Save or display the image
  cv::imwrite("output.png", image);
  cv::imshow("Output Image", image);
  // cv::waitKey(0);

  
  return 0;
}