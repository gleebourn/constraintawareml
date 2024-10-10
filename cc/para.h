#include "lens.h"

template<typename T>
concept para = arrow<T>&&
               arrow<typename T::infer>&&
               requires{
  typename T::param;
  typename T::infer;
  std::is_same_v<typename T::infer::dom,std::tuple<typename T::param,typename T::dom>>;
  std::is_same_v<typename T::infer::cod,typename T::cod>;
};
