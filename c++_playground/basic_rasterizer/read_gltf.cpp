#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

int parse_gltf(){
    cgltf_options options = {};
    cgltf_data* data = NULL;
    cgltf_result result = cgltf_parse_file(&options, "/Users/arnaudstiegler/gaussian-splat/c++_playground/basic_rasterizer/artifacts/1995_-_mclaren_f1_gtr_rigged__mid-poly/scene.gltf", &data);
    if (result == cgltf_result_success){
        /* TODO make awesome stuff */
        std::cout << "Mesh count " << data->meshes_count << std::endl;
        int count_primitives = 0;
        for (int i = 0; i < data->meshes_count; i++){
            cgltf_mesh mesh = data->meshes[i];
            std::cout << "Primitive count " << mesh.primitives_count << std::endl;
            for (int j = 0; j < mesh.primitives_count; j++){
                cgltf_primitive primitive = mesh.primitives[j];
                for (cgltf_size i = 0; i < primitive.attributes_count; ++i) {
                    cgltf_attribute attribute = primitive.attributes[i];
                    if (attribute.type == cgltf_attribute_type_position) {
                            cgltf_accessor* accessor = attribute.data;
                        }
                }
                //     cgltf_attribute* attribute = &primitive->attributes[i];
                //     if (attribute->type == CGLTF_ATTRIBUTE_TYPE_POSITION) {
                //         cgltf_accessor* accessor = attribute->data;
        // Now use accessor to read the vertex position data
                //     }
                // }
                count_primitives += 1;
            }
        }

        cgltf_free(data);
        
    }
    /*
    Each mesh has a single primitive, and there are 204 meshes so 204 primitives total.
    */
    return 0;
}