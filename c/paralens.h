#include<stdio.h>


typedef struct para_lens{
  void*(*fd)(struct para_lens*self);
  void*(*bk)(struct para_lens*self);
  void* p;
  void* ps;
  void* x;
  void* y;
  void* xs;
  void* ys;
}para_lens;

void f(para_lens*a){
  a->fd(a);
}

void b(para_lens*a){
  a->bk(a);
}

void pass_fd(para_lens*a,para_lens*b){
  b->x=a->y;
}

void pass_bk(para_lens*a,para_lens*b){
  a->ys=b->xs;
}

void composition_fd(para_lens**l,int n){
  int i=0;
  while(1){
    f(l[i]);
    if(i=n-1)break;
    pass_fd(l[i],l[i+1]);
    i++;
  }
}

void composition_bk(para_lens**l,int n){
  composition_fd(l,n);
  int i=n-1;
  while(1){
    b(l[i]);
    if(i==0)break;
    pass_bk(l[i],l[i-1]);
    i--;
  }
}
