#include <json/json.h>
#include <iostream>
#include <fstream>

int main(){
    // First define our filname, would probbably be better to prompt the user for one
    const std::string& gltfFilename = "example.gltf"

    // open the gltf file
    std::ifstream jsonFile(gltfFilename, std::ios::binary);

    // parse the json so we can use it later
    Json::Value json;

    try{
        jsonFile >> json;
    }catch(const std::exception& e){
        std::cerr << "Json parsing error: " << e.what() << std::endl;
    }
    jsonFile.close();

    // Extract the name of the bin file, for the sake of simplicity I'm assuming there's only one
    std::string binFilename = json["buffers"][0]["uri"].asString();

    // Open it with the cursor at the end of the file so we can determine it's size,
    // We could techincally read the filesize from the gltf file, but I trust the file itself more
    std::ifstream binFile = std::ifstream(binFilename, std::ios::binary | std::ios::ate);

    // Read file length and then reset cursor
    size_t binLength = binFile.tellg();
    binFile.seekg(0);


    std::vector<char> bin(binLength);
    binFile.read(bin.data(), binLength);
    binFile.close();



    // Now that we have the files read out, let's actually do something with them
    // This code prints out all the vertex positions for the first primitive

    // Get the primitve we want to print out: 
    Json::Value& primitive = json["meshes"][0]["primitives"][0];


    // Get the accessor for position: 
    Json::Value& positionAccessor = json["accessors"][primitive["attributes"]["POSITION"].asInt()];


    // Get the bufferView 
    Json::Value& bufferView = json["bufferViews"][positionAccessor["bufferView"].asInt()];


    // Now get the start of the float3 array by adding the bufferView byte offset to the bin pointer
    // It's a little sketchy to cast to a raw float array, but hey, it works.
    float* buffer = (float*)(bin.data() + bufferView["byteOffset"].asInt());

    // Print out all the vertex positions 
    for (int i = 0; i < positionAccessor["count"].asInt(); ++i){
        std::cout << "(" << buffer[i*3] << ", " << buffer[i*3 + 1] << ", " << buffer[i*3 + 2] << ")" << std::endl;
    }

    // And as a cherry on top, let's print out the total number of verticies
    std::cout << "vertices: " << positionAccessor["count"].asInt() << std::endl;

    return 0;
}