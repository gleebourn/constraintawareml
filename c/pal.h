#include "paralens.h"

void pal_fd(para_lens l){
  float *t=(float*)l.y;
  *t=0;
  for(int i=0;i<para_lens.p.len;i++)
    *t+=l.p.w[i]*l.x[i];
}

void pal_bk(para_lens l){

}
