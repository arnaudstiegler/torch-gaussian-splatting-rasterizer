# include <ostream>
# include <iostream>

class Animal {
    private:
        bool is_female;
        int age;
    
    public:
        float weight;
        // That's the constructor with initializer list for each attribute
        Animal(bool is_female, int age, float weight): is_female(is_female), age(age), weight(weight){
        };
        // An alternative way to implement the constructor with the "this" point
        // Animal(bool is_female, int age, float weight){
        //     this->is_female = is_female;
        //     this->age = age;
        //     this->weight = weight;
        // }

        // We are overloading the << function for the class
        friend std::ostream& operator<<(std::ostream& stream, const Animal& animal);
};

std::ostream& operator<<(std::ostream& stream, const Animal& animal){
    stream << "Age: " << animal.age << " Weight: " << animal.weight;
    return stream; 
};

class Lion: public Animal {
    public:
        // In C++, you don't automatically inherit the constructor from the parent class
        Lion(bool is_female, int age, float weight): Animal(is_female, age, weight){};
        friend std::ostream& operator<<(std::ostream& stream, const Lion& lion);
};

std::ostream& operator<<(std::ostream& stream, const Lion& lion){
    stream << static_cast<const Animal&>(lion);
    stream << "Species: Lion";
    return stream;
}

int main(){
    Animal animal = Animal(true, 10, 40);
    Lion lion = Lion(false, 5, 20);

    std::cout << animal << std::endl;
    std::cout << lion << std::endl;
    return 0;
}