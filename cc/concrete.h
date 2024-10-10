#include<type_traits>
#include<concepts>
#include<functional>
#include<print>

template<typename T>
concept arrow = requires{
  typename T::dom;
  typename T::cod;
};

template<typename T>
concept composed_pair=arrow<T>&&
                      arrow<typename T::fst>&&
                      arrow<typename T::scnd>&&
                       requires{
  std::is_same_v<typename T::fst::cod,typename T::scnd::dom>;
  std::is_same_v<typename T::fst::dom,typename T::dom>;
  std::is_same_v<typename T::scnd::cod,typename T::cod>;
};

template<typename F,typename G>
concept composable = arrow<F>&&arrow<G>&&requires{
  std::is_same_v<typename F::cod,typename G::dom>;
  false;
};

template<typename T>
concept concrete = requires(T t){
  std::is_same_v<typename T::fn,std::function<typename T::cod(typename T::dom)>>;//check has associated fn typename
  std::is_same_v<decltype(t.underlying),typename T::fn>;
};

template<typename F,typename G>
concept concrete_composable=composable<F,G>&&concrete<F>&&concrete<G>&&requires{false;};

template<typename A,typename B>
struct atomic_concrete{
  typedef A dom;
  typedef B cod;
  typedef std::function<B(A)> fn;
  fn underlying;
  atomic_concrete(fn und):underlying(und){}
  //static_assert(concrete<atomic_concrete<A,B>>);
};

template<typename F,typename G>
requires concrete_composable<F,G>
struct composed_concrete{
  typedef F fst;
  typedef G scnd;
  fst first;
  scnd second;
  typedef typename F::dom dom;
  typedef typename G::cod cod;
  static_assert(std::is_same_v<typename F::cod,typename G::dom>);
  typedef typename std::function<typename G::cod(typename F::dom)> fn;
  composed_concrete(F f,G g):first(f),second(g){}
  fn underlying=[&](F::dom x){
    return second.underlying(first.underlying(x));
  };
  //static_assert(concrete<composed_concrete<F,G>>);
};

template<typename F,typename G> requires concrete_composable<F,G>
composed_concrete<F,G> operator*(G g,F f){
  return composed_concrete<F,G>(f,g);
};
