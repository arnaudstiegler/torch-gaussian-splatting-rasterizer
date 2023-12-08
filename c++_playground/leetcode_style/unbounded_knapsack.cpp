 # include <vector>
 # include <iostream>

int W = 100;
std::vector<int> val = {10, 30, 20};
std::vector<int> w = {5,10,15};

int unbounded_knapsack(int max_w){
    if (max_w == 0){
        return 0;
    }
    else {
        int max_val = 0;
        for (int i=0; i < w.size(); i++){
            if (max_w >= w[i]){
                int current_iter = unbounded_knapsack(max_w - w[i]) + val[i];
                if ( current_iter > max_val){
                    max_val = current_iter;
                }
            }
        }
        return max_val;
    }
}

int main(){
    std::cout << unbounded_knapsack(W) << std::endl;
    return 0;
}