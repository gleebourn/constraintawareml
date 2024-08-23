#include<fstream>
#include<iostream>
#include<ios>
#include<math.h>
#include<optional>
#include<array>
#include<format>

template <typename P,typename X,typename Y>
class Learner {
  public:
    virtual std::tuple<P,std::optional<X>,std::optional<Y>>
      process (P p,std::optional<X> x,std::optional<Y> y)const{
      std::cout<<"F";
      return std::tuple(p,x,y); //Trivial learner
    };
};

template <typename P,typename Q,typename X,typename Y,typename Z>
class ComposedLearner : public Learner<std::tuple<Q,P>,X,Z> {
  private:
    const Learner<P,X,Y>& first;
    const Learner<Q,Y,Z>& second;
  public:
    ComposedLearner(const Learner<Q,Y,Z>& a,const Learner<P,X,Y>& b):first(b),second(a){};

    std::tuple<std::tuple<Q,P>,std::optional<X>,std::optional<Z>>
      process(std::tuple<Q,P> pq,std::optional<X> x,std::optional<Z> z)const{
      auto xy_inference = first.process(std::get<1>(pq),x,std::nullopt);
      auto yz_inference = second.process(std::get<0>(pq),
                                         std::get<2>(xy_inference),z);

      auto xy_back_inference = first.process(std::get<0>(xy_inference),
                                             std::get<1>(xy_inference),
                                             std::get<1>(yz_inference));

      return std::tuple(std::tuple(std::get<0>(yz_inference),
                                   std::get<0>(xy_back_inference)),
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
    TensoredLearner(Learner<P,W,X>& a,Learner<Q,Y,Z>& b): left(a),right(b){};

     std::tuple<std::tuple<P,Q>,
                std::optional<std::tuple<W,Y>>,
                std::optional<std::tuple<std::optional<X>,std::optional<Z>>>>
      process(std::tuple<P,Q> pq,
              std::optional<std::tuple<std::optional<W>,std::optional<Y>>> wy,
              std::optional<std::tuple<std::optional<X>,std::optional<Z>>> xz)const{

      auto rwy= xz.value_or(std::tuple(std::nullopt,std::nullopt));
      auto rxz= xz.value_or(std::tuple(std::nullopt,std::nullopt));

      auto wx_inference = left.process(std::get<0>(pq),std::get<0>(rwy),std::get<0>(rxz));
      auto yz_inference = right.process(std::get<1>(pq),std::get<1>(rwy),std::get<1>(rxz));
      return std::tuple(std::tuple(std::get<0>(wx_inference),std::get<0>(yz_inference)),
                        std::tuple(std::get<1>(wx_inference),std::get<1>(yz_inference)),
                        std::tuple(std::get<2>(wx_inference),std::get<2>(yz_inference)));
    }
};

template <typename P,typename Q,typename X,typename Y,typename Z>
auto operator*(const Learner<P,Y,Z> &a,const Learner<Q,X,Y> &b){
  return ComposedLearner(a,b);
}

template <typename P,typename Q,typename W,typename X,typename Y,typename Z>
auto operator+(Learner<P,W,X> a,Learner<Q,Y,Z> b){
  return TensoredLearner(a,b);
}

template<typename F,typename T,int N>
class BiasLearner:public Learner<std::tuple<F,std::array<T,N>>,
                                 std::array<T,N>,
                                 std::array<T,N>>{
  public:
    BiasLearner(){};
    std::tuple<std::tuple<F,std::array<T,N>>,
               std::optional<std::array<T,N>>,
               std::optional<std::array<T,N>>> process(
    std::tuple<F,std::array<T,N>>fp,
    std::optional<std::array<T,N>> x,
    std::optional<std::array<T,N>> y)const{
      std::array<T,N>po;
      std::array<T,N>xo;
      std::array<T,N>yo;
      auto f=std::get<0>(fp);
      auto p=std::get<1>(fp);
      auto rx=x.value_or(std::array<T,N>());
      auto ry=y.value_or(rx);
      for(int i=0;i<N;i++){
        yo[i]=rx[i]+p[i];
        xo[i]=ry[i]-p[i];
        po[i]=(T)f*p[i]+(1-f)*(ry[i]-rx[i]);
      }
      return std::tuple(std::tuple(f,po),std::optional(xo),std::optional(yo));
    }
};

template<typename F,typename T,int M,int N>
class ArtificialUnbiasedReluNeuron:
public Learner<std::tuple<F,std::array<std::array<T,M>,N>>,
               std::array<T,M>,
               std::array<T,N>>{
  public:
    std::tuple<std::tuple<F,std::array<std::array<T,M>,N>>,
               std::optional<std::array<T,M>>,
               std::optional<std::array<T,N>>>
    process(std::tuple<F,std::array<std::array<T,M>,N>> ep,
            std::optional<std::array<T,M>> x,
            std::optional<std::array<T,N>>y)const{

      std::array<T,M> xo;
      std::array<T,N> yo;
      F energy=std::get<0>(ep);
      auto p=std::get<1>(ep);
      auto rx=x.value_or(std::array<T,M>());
      for(int i=0;i<N;i++){
        for(int j=0;j<M;j++)
          yo[i]+=p[i][j]*rx[j];

        if(yo[i]<0) //ReLu
          yo[i]=0;
      }

      std::array<std::array<T,M>,N> po;
      auto ry=y.value_or(std::array<T,N>());
      for(int i=0;i<M;i++){
        xo[i]=rx[i];
        for(int j=0;j<N;j++){
          T tmp=(yo[j]-ry[j])*yo[j];
          po[j][i]=int(po[j][i]-energy*tmp*rx[i]); //Gradient descent
          xo[i]-=tmp*p[j][i];
        }
      }

      return std::tuple(std::tuple(energy,po),std::optional(xo),std::optional(yo));
    };
};

//template<typename A,typenam B,typenam C,typename T,char M1,char M2,char N1,char N2>
//class FlattenedLearner: Learner{
//  private:
//    unflattenedLearner
//  public:
//      FlattenedLearner(
//}

#define HEADER_LENGTH 18

#define N_CHARS 62
#define SAMPLES_PER_CHAR 55
#define N_PICS N_CHARS*SAMPLES_PER_CHAR
#define PIC_WIDTH 32
#define PIC_HEIGHT 24
#define PIC_SIZE PIC_WIDTH*PIC_HEIGHT
#define EPOCHS 10

auto load_data(){
  std::array<std::array<int8_t,PIC_SIZE>,N_PICS> X;
  std::array<int8_t,N_PICS> y;
  for(int i=0;i<N_CHARS;i++){
    for(int j=0;j<SAMPLES_PER_CHAR;j++){
      auto fn=std::format("/home/glee/Downloads/archive/Img_resized/img{:03}-{:03}.tga",i+1,j+1);
      std::basic_ifstream<char> is(fn,std::ifstream::in);
      if(!is.is_open()){std::cout<<"Oh dear!!!\n  Opening failed..\n";exit(1);}

      is.seekg(HEADER_LENGTH,std::ios_base::beg);

      for(int row=0;row<PIC_HEIGHT;row++){
        //for(int cn=0;cn<3;cn++)
        for(int col=0;col<PIC_WIDTH;col++)
          X[i*SAMPLES_PER_CHAR+j][row*PIC_WIDTH+col]=(int8_t)is.get();

      }

      y[i * SAMPLES_PER_CHAR + j] = i;
    }
  }
  return std::tuple(X,y);
}

template<int M,int N>
void print_img(std::array<int8_t,M*N>img){
  static int l=M*N;
  for(int i=0;i<l;i+=M){
    for(int j=0;j<M;j++)
      if(img[i+j])
        std::cout<<(((int)img[i+j]%10+10)%10);
      else
        std::cout<<' ';
    std::cout<<"\n";
  }
}

#define L_IN 768
#define L_1 64
#define L_2 16

template<typename T,int N>
std::array<T,N> one_hot(int t){
  auto ret=std::array<T,N>();
  ret[t]=T(1);
  return ret;
}

int main(){

  auto f=ArtificialUnbiasedReluNeuron<float,int8_t,L_IN,L_1>();
  std::tuple<float,std::array<std::array<int8_t,L_IN>,L_1>>
    pf=std::tuple(.1,std::array<std::array<int8_t,L_IN>,L_1>());

  auto g=ArtificialUnbiasedReluNeuron<float,int8_t,L_1,L_2>();
  std::tuple<float,std::array<std::array<int8_t,L_1>,L_2>>
    pg=std::tuple(.2,std::array<std::array<int8_t,L_1>,L_2>());
  
  auto h=ArtificialUnbiasedReluNeuron<float,int8_t,L_2,N_CHARS>();
  std::tuple<float,std::array<std::array<int8_t,L_2>,N_CHARS>>
    ph=std::tuple(.3,std::array<std::array<int8_t,L_2>,N_CHARS>());

  auto b=BiasLearner<float,int8_t,N_CHARS>();
  std::tuple<float,std::array<int8_t,N_CHARS>>
    pb=std::tuple(.05,std::array<int8_t,N_CHARS>());

  auto multi_layer=b*h*g*f;

  auto p0=std::tuple(std::tuple(std::tuple(pb,ph),pg),pf);
  auto x0=std::optional(std::array<int8_t,L_IN>());
  auto y0=std::optional(std::array<int8_t,N_CHARS>());

  std::cout<<"\n";
  auto d=load_data();
  auto X=std::get<0>(d);
  auto y=std::get<1>(d);
  std::cout<<"\n\n\n\n";
  print_img<PIC_WIDTH,PIC_HEIGHT>(X[5]);
  
  for(int e=0;e<EPOCHS;e++){
    for(int i=0;i<N_PICS;i++){
      p0=std::get<0>(multi_layer.process(p0,X[i],one_hot<int8_t,N_CHARS>(y[i])));
    };
    std::cout<<"Epoch over!\n";
  }
  std::cout<<"\nTesting (of course we have likely overfit if this even works at all)\n";
  int loss=0;
  for(int i=0;i<N_PICS;i++){
    auto y_pred=std::get<2>
      (multi_layer.process(p0, X[i], std::nullopt)).value();
    for(int j=0;j<N_CHARS;j++){//Check we have the one-hot encoding we want, this is L1 loss I guess
      loss+=(y_pred[j]!=(y[i]==j));
    }
  }
  std::cout<<loss;
  std::cout<<"\n";
  std::cout<<"\n";
  
  return 0;
}
