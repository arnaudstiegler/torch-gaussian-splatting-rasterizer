 # include <vector>
 # include <iostream>
 # include <unordered_map>
 

int W = 100;
std::vector<int> val = {10, 30, 20};
std::vector<int> w = {5,10,15};

int unbounded_knapsack(int max_w, std::unordered_map<int, int> memoization_dict){
    if (max_w == 0){
        return 0;
    }
    else {
        int max_val = 0;
        for (int i=0; i < w.size(); i++){
            if (max_w >= w[i]){
                // Notice how weird the check is to find whether the key is in the memoization_dict
                if (memoization_dict.find(max_w - w[i]) != memoization_dict.end()){
                    return memoization_dict[max_w - w[i]];
                }
                else {
                    int current_iter = unbounded_knapsack(max_w - w[i], memoization_dict) + val[i];
                    memoization_dict[max_w - w[i]] = current_iter;
                    if ( current_iter > max_val){
                        max_val = current_iter;
                    }
                }
            }
        }
        return max_val;
    }
}

int main(){
    std::unordered_map<int, int> memoization_dict;
    std::cout << unbounded_knapsack(W, memoization_dict) << std::endl;
    return 0;
}