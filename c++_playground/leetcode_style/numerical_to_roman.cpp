# include <map>
# include <string>
# include <iostream>
# include <vector>
# include <cassert>


std::vector<std::pair<int, std::string>> symbol_mapping = {
    {1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"},
    {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"},
    {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}
};

std::string convert_to_roman(int num){
    std::string roman = "";
    while (num > 0) {
        for (const auto& pair: symbol_mapping){
            if (num >= pair.first) {
                roman = roman + pair.second;
                num -= pair.first;
                break;
            }
        }
    }
    return roman;
};

int main(){
    std::string test_one = convert_to_roman(19);
    std::string test_two = convert_to_roman(1994);

    std::cout << test_one << std::endl;
    std::cout << test_two << std::endl;

    assert(test_one == "XIX");
    assert(test_two == "MCMXCIV");

    return 0;
}