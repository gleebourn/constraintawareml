#include <vector>
#include <iostream>

int main(){
  std::tuple<int,int> a;
  a=std::tuple(4,5);
  std::cout<<std::get<0>(a)<<"\n";
  std::cout<<std::get<1>(a)<<"\n";
  std::cout<<"Hi there!\n";
  return 0;
}
