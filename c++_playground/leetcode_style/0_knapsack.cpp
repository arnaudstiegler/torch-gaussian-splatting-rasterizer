# include <vector>
# include <iostream>
# include <unordered_map>

std::vector<int> profit = {59, 63, 15, 43, 75};
std::vector<int> weight = {16, 13, 10, 19, 6};
int W = 38;

// You have to create a custom hash for the unordered_map to work
struct pair_hash {
    inline std::size_t operator()(const std::pair<int, int> & v) const {
        return std::hash<int>()(v.first) ^ std::hash<int>()(v.second);
    }
};

int knapsack(int w, int length){
    std::unordered_map<std::pair<int, int>, int, pair_hash> memoization_dict;
    
    // Pre-fill for current_index = 0
    for (int current_weight = 0; current_weight <= w; current_weight++){
        // Note the ternary operator here
        memoization_dict[{current_weight, 0}] = (current_weight >= weight[0]) ? profit[0] : 0;
    }

    // Doing it with dynamic programming (bottom up)
    for (int current_index = 1; current_index < length; current_index ++){
        for (int current_weight = 0; current_weight <= w; current_weight++){
            if (current_weight - weight[current_index] < 0){
                // can't add the current weight any way
                memoization_dict[{current_weight, current_index}] = memoization_dict[{current_weight, current_index - 1}];
            }
            else {
                int add_weight_val = memoization_dict[{current_weight - weight[current_index], current_index - 1}] + profit[current_index];
                int dont_add_weight_val = memoization_dict[{current_weight, current_index - 1}];
                memoization_dict[{current_weight, current_index}] = std::max(add_weight_val, dont_add_weight_val);
            }
        }
    }
    return memoization_dict[{w, length-1}];
}


int main(){
    std::cout << knapsack(W, profit.size()) << std::endl;
    return 0;
}