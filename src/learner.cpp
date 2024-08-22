#include<iostream>
#include<math.h>
#include <optional>
#include <memory>

template <typename P,typename X,typename Y>
class Learner {
  public:
    virtual std::tuple<P,std::optional<X>,std::optional<Y>>
      process(P p,std::optional<X> x,std::optional<Y> y){
      return std::tuple(p,x,y); //Trivial learner
    };
};

template <typename P,typename Q,typename X,typename Y,typename Z>
class ComposedLearner : public Learner<std::tuple<P,Q>,X,Z> {
  private:
    Learner<P,X,Y> first;
    Learner<Q,Y,Z> second;
  public:
    ComposedLearner(Learner<Q,Y,Z> const& a,Learner<P,X,Y> const& b):first(b),second(a){};

    std::tuple<std::tuple<P,Q>,std::optional<X>,std::optional<Z>>
      process(std::tuple<P,Q> pq,std::optional<X> x,std::optional<Z> z){
      auto xy_inference = first.process(std::get<0>(pq),x,std::nullopt);
      auto yz_inference = second.process(std::get<1>(pq),
                                         std::get<2>(xy_inference),z);
      auto q_new=std::get<0>(yz_inference);
      auto xy_back_inference = first.process(std::get<0>(xy_inference),
                                             std::get<1>(xy_inference),
                                             std::get<1>(yz_inference));
      return std::tuple(std::tuple(std::get<0>(xy_back_inference),q_new),
                        std::get<1>(xy_back_inference),
                        std::get<2>(yz_inference));
    }
};

template <typename P,typename Q,typename W,typename X,typename Y,typename Z>
class TensoredLearner : public Learner<std::tuple<P,Q>,std::tuple<W,Y>,std::tuple<X,Z>> {
  private:
    Learner<P,W,X> left;
    Learner<Q,Y,Z> right;
  public:
    TensoredLearner(Learner<P,W,X> const& a,Learner<Q,Y,Z> const& b): left(a),right(b){};

    std::tuple<std::tuple<P,Q>,
               std::optional<std::tuple<std::optional<W>,std::optional<Y>>>,
               std::optional<std::tuple<std::optional<X>,std::optional<Z>>>>
      process(std::tuple<P,Q> pq,
              std::optional<std::tuple<std::optional<W>,std::optional<Y>>> wy,
              std::optional<std::tuple<std::optional<X>,std::optional<Z>>> xz){

      auto rwy= wy.value_or(std::tuple(std::optional<W>(),std::optional<Y>()));
      auto rxz= xz.value_or(std::tuple(std::optional<X>(),std::optional<Z>()));

      auto wx_inference = left.process(std::get<0>(pq),std::get<0>(rwy),std::get<0>(rxz));
      auto yz_inference = right.process(std::get<1>(pq),std::get<1>(rwy),std::get<1>(rxz));
      return std::tuple(std::tuple(std::get<0>(wx_inference),std::get<0>(yz_inference)),
                        std::tuple(std::get<1>(wx_inference),std::get<1>(yz_inference)),
                        std::tuple(std::get<2>(wx_inference),std::get<2>(yz_inference)));
    }
};

template <typename P,typename Q,typename X,typename Y,typename Z>
auto operator*(Learner<P,Y,Z> const& a,Learner<Q,X,Y> const& b){
  return ComposedLearner(a,b);
}

template <typename P,typename Q,typename W,typename X,typename Y,typename Z>
auto operator+(Learner<P,W,X>const& a,Learner<Q,Y,Z> const& b){
  return TensoredLearner(a,b);
}

int main(){
  auto f=Learner<bool,int,float>();
  auto g=Learner<bool,float,float>();
  auto h=Learner<bool,std::tuple<float,float>,float>();
  auto test1=f+g;
  auto x=test1.process(std::tuple(false,false),std::tuple(std::optional(5),std::optional(23.4)),
                       std::tuple(std::optional(2.4),std::optional(5.2)));
  //std::cout<<std::get<0>(std::get<1>(x));
  std::cout<<"\n";
  auto test2=g*f;
  auto test3=h*(f+g);
  return 0;
}
