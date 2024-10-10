#include "concrete.h"

template<typename F,typename B>
concept compatible_lens_data = arrow<F>&&arrow<B>&&requires{
  std::is_same_v<typename F::dom,std::tuple_element<0,typename B::dom>>;
};

template<typename T>
concept lens_arrow = arrow<T>&&
        compatible_lens_data<typename T::fd,typename T::bck>&&
        requires{
  std::is_same_v<typename T::dom,
                 std::tuple<typename T::fd::dom,
                            std::tuple_element<0,typename T::bk::dom>>>;
  std::is_same_v<typename T::cod,
                 std::tuple<typename T::fd::cod,
                            std::tuple_element<1,typename T::bk::dom>>>;
};

template<typename T>
concept concrete_lens_arrow=lens_arrow<T>&&concrete<typename T::fd>&&concrete<typename T::bk>;


template<typename F,typename G>
concept concrete_composable_lenses=concrete_lens_arrow<F>&&concrete_lens_arrow<G>&&composable<F,G>;

template<typename T>
requires lens_arrow<T>
struct atomic_concrete_lens{
  T::fd forward;
  T::bck back;
  atomic_concrete_lens(T::fd f,T::bck b):forward(f),back(b){}
};

template<typename F,typename G>
requires concrete_composable_lenses<F,G>
struct composed_concrete_lenses{
};
