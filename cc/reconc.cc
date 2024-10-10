#include<functional>

//template<typename P,typename Ps,typename A,typename As,typename B,typename Bs>
template<typename S,typename T>
struct lens{
  typedef typename S::P P;
  typedef typename S::Ps Ps;
  typedef typename S::A A;
  typedef typename S::As As;
  typedef typename T::A B;
  typedef typename T::As Bs;
  typedef std::tuple<A,As> src;
  typedef std::tuple<B,Bs> tgt;
  typedef std::tuple<P,Ps> par;
  typedef std::function<B(std::tuple<P,A>)> fwd;
  typedef std::function<std::tuple<As,Ps>(std::tuple<std::tuple<P,A>,Bs>)> bck;
  fwd forward;
  bck back;
  std::tuple<P,A> in_S;
  Bs in_T;
  lens(fwd f,bck b):forward(f),back(b){}

  B go_forward(){
    return fwd(in_S);
  }

  std::tuple<As,Ps> go_back(){
    return bck(std::make_tuple(in_S,in_T));
  }
};


template<typename A,typename ...Z>
struct ret_val{
  typedef A ret;
};

template<typename A,typename B,typename...Z>
struct ret_val<A,B,Z...>{
  typedef ret_val<B,Z...>::ret ret;
};

template<typename ...Z>
int n_terms=0;

template<typename A,typename ...Z>
int n_terms=n_terms<Z...>+1;

template<typename A>
struct composition{
  int n=0;
  lens<A,B>fst;
  composition(lens<A,B>f):fst(f){}
};

template<typename A,typename B,typename...Z>
struct composition<A,B,Z...>{
  int n=n_terms<B,Z...>;
  lens<A,B> fst;
  composition<B,Z...> rst;
};

int main(){
}
