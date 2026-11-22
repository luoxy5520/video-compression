from pathlib import Path
import shutil,re
ROOT=Path.home()/'lmp-e1-e2-followup-20261121';src=Path.home()/'a_new48_20261107/kit/linux/JM';dst=ROOT/'jm-profile'
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('obj','*.o','*.exe'),dirs_exist_ok=True)
p=dst/'lencod/src/mv_search.c';s=p.read_text();s='''#include <time.h>
#include <stdlib.h>
#include <stdio.h>
unsigned long long lmp_ie=0,lmp_se=0;
static double lmp_it=0,lmp_st=0;
static double lmp_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void lmp_report(void){fprintf(stderr,"LMP_PROFILE {\\"integer_s\\":%.9f,\\"subpixel_s\\":%.9f,\\"integer_evals\\":%llu,\\"subpixel_evals\\":%llu}\\n",lmp_it,lmp_st,lmp_ie,lmp_se);}
static int lmp_ready=0;
'''+s
for call,acc in [('min_mcost = currMB->IntPelME (currMB, &pred, mv_block, min_mcost, lambda_factor[F_PEL]);','lmp_it'),('min_mcost =  currMB->SubPelME (currMB, &pred, mv_block, min_mcost, lambda_factor);','lmp_st')]:
 assert s.count(call)==1
 s=s.replace(call,'{ double lmp_t; if(!lmp_ready){atexit(lmp_report);lmp_ready=1;} lmp_t=lmp_now(); '+call+' '+acc+' += lmp_now()-lmp_t; }')
p.write_text(s)
for p in (dst/'lencod/src').glob('*.c'):
 if p.name=='mv_search.c':continue
 s=p.read_text();changed=False
 for fn,counter in [('computePredFPel','lmp_ie'),('computePredHPel','lmp_se'),('computePredQPel','lmp_se')]:
  s,n=re.subn(r'(mv_block->'+fn+r'\s*\()\s*(ref_picture)',r'\1(++'+counter+r', \2)',s);changed|=bool(n)
 if changed:p.write_text('extern unsigned long long lmp_ie,lmp_se;\n'+s)
