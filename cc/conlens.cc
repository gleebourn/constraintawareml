#include<functional>


template<typename A,typename As,typename B,typename Bs>
struct con_lens{
  std::function<B(A)> forward;
  std::function<As(std::tuple<A,Bs>)> back;
  con_lens(std::function<B(A)>f,std::function<As(std::tuple<A,Bs>)>b):forward(f),back(b){}
};

template<typename P,typename Ps,typename A,typename As,typename B,typename Bs>
struct para_lens{
  con_lens<std::tuple<A,P>,std::tuple<As,Ps>,B,Bs> underlying;
};

//Compose two parametric lenses
template<typename P,typename Ps,typename Q,typename Qs,typename A,typename As,typename B,typename Bs,typename C,typename Cs>
para_lens<std::tuple<P,Q>,std::tuple<Ps,Qs>,A,As,B,Bs>operator*(para_lens<Q,Qs,B,Bs,C,Cs> g,para_lens<P,Ps,A,As,B,Bs> f){
  auto fd=[&f,&g](std::tuple<A,P,Q>apq){return g.forward(std::make_tuple(f.forward(std::make_tuple(std::get<0>(apq),
                                                                                              std::get<1>(apq)),
                                                                    std::get<2>(apq))));};
  auto bk=[&f,&g](std::tuple<std::tuple<A,P,Q>,Cs>apq_c){
    return f.backward(std::make_tuple(std::make_tuple(std::get<0>(std::get<0>(apq_c)),
                                                      std::get<1>(std::get<0>(apq_c))),
                                      g.backward(std::make_tuple(std::make_tuple(f.forward(std::make_tuple(std::get<0>(std::get<0>(apq_c)),
                                                                                                           std::get<1>(std::get<0>(apq_c)))),
                                                                                 std::get<2>(std::get<0>(apq_c))),
                                                                 std::get<1>(apq_c)))));};
  return con_lens(fd,bk);
}

template<typename F,typename ...>



template<typename A,typename As,typename B,typename Bs,typename C,typename Cs>
con_lens<A,As,C,Cs>operator*(con_lens<B,Bs,C,Cs> g,con_lens<A,As,B,Bs> f){
  return con_lens(std::function([&f,&g](A x){return g.forward(f.forward(x));}),
                       std::function([&f,&g](std::tuple<A,Cs> xz){return f.back(std::make_tuple(std::get<0>(xz),
                                                                                                g.back(std::make_tuple(f.forward(std::get<0>(xz)),
                                                                                                       std::get<1>(xz)))));}));
}


int main(){
  std::function af([](float x){return int(x*x);});
  std::function ab([](std::tuple<float,int>xy){return std::get<0>(xy)+std::get<1>(xy)>3;});

  std::function bf([](int n){return n<20;});
  std::function bb([](std::tuple<int,bool>bx){return int(std::get<0>(bx)*std::get<1>(bx));});

  auto a=con_lens(af,ab);
  auto b=con_lens(bf,bb);
  //auto c=a*b;
  auto d=b*a;

}
