#include <iostream>
#include <vector>

int sum(int a, int b) {
    return a + b;
}

int find_min(int* array, int size){
    int current_min = array[0];

    // The way to get the size of the array is antiquated and due to
    // using a basic array (and not an std::array)
    for (int i = 0; i < size; i++){
        if (current_min > array[i]) {
            current_min = array[i];
        }
    }
    return current_min;
};

void create_arrays(int size) {
    // Creates an empty with a fixed size
    int array_a[5];

    // Creates an array with a fixed size and populates it
    int array_b[5] = {1,2,3,4,5};

    int min = find_min(array_b, 5);

    std::cout << min;

    // Creates an array with "automatic" size deduction
    int array_c[] = {1,2,3,4,5};

    // Creates an array with a dynamic size
    // ie. the size is determined at run time
    int* array_d = new int[size];
    delete[] array_d;

    // Uses standard library std::vector
    // Automatically manages memory and can be dynamically resized
    std::vector<int> array_e = {1,2,3,4,5};
}

int main() {
    // To print the C++ version used by the compiler
    std::cout << __cplusplus;
    create_arrays(sum(1,2));
    return 0;
}