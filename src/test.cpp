#include <vector>
#include <iostream>

int main(){
  int8_t a=(int8_t)128;
  int8_t b=-127;
  std::cout<<(int)a;
  std::cout<<"\n";
  std::cout<<(int)b;
  std::cout<<"\n";
  std::cout<<(int)(int8_t)(a+b);
  std::cout<<"\n";
  return 0;
}
