from pathlib import Path
import shutil
R=Path.home()/'lmp-e1-e2-followup-20261121';dst=R/'jm-d3';shutil.copytree(R/'jm-profile',dst,dirs_exist_ok=True)
p=dst/'lencod/src/me_epzs_int.c';s=p.read_text();s='''#include <stdio.h>
#include <stdlib.h>
static short lmp_candidates[16*30*52*8*2];
static int lmp_loaded=0;
static int lmp_mode(void){char *s=getenv("LMP_MODE");return s?atoi(s):0;}
static int lmp_load(void){FILE *f;char *s;if(lmp_loaded)return lmp_loaded>0;s=getenv("LMP_CANDIDATES");if(!s)return 0;f=fopen(s,"rb");if(!f)return 0;lmp_loaded=fread(lmp_candidates,sizeof(short),16*30*52*8*2,f)==16*30*52*8*2?1:-1;fclose(f);return lmp_loaded>0;}
'''+s
marker='  // Additional threshold for ref>0';idx=s.index(marker)
code='''  /* Cached latent proposal; evaluated against the LIVE reconstructed reference.
     Original predictor remains intact. No historical MV labels read here. */
  if (blocktype==1 && list==0 && ref==0 && p_Vid->frame_no>=1 && p_Vid->frame_no<=16)
  {
    int mode=lmp_mode(), used=0, improved=0;
    if(mode>=2 && lmp_load())
    {
      int ix=(((p_Vid->frame_no-1)*30+currMB->pix_y/16)*52+currMB->pix_x/16)*16;
      MotionVector proposal; distblk cost;
      proposal.mv_x=lmp_candidates[ix];proposal.mv_y=lmp_candidates[ix+1];
      if (iabs(proposal.mv_x-mv->mv_x)<=searchRange->max_x && iabs(proposal.mv_y-mv->mv_y)<=searchRange->max_y)
      {
        MotionVector padded=pad_MVs(proposal,mv_block);
        cost=mv_cost(p_Vid,lambda_factor,&padded,&pred);
        cost += mv_block->computePredFPel(ref_picture,mv_block,DISTBLK_MAX-cost,&padded);used=1;
        if(cost<min_mcost){min_mcost=cost;tmp=proposal;improved=1;}
      }
    }
    /* Pre-search early stopping; mode 1 is the same native budget control. */
    if(mode==1 || (mode==3 && improved))stopCriterion*=2;
    fprintf(stderr,"LMP_BLOCK %d %d %d %d %d %d\\n",p_Vid->frame_no,currMB->pix_x,currMB->pix_y,mode,used,improved);
  }
'''
s=s[:idx]+code+s[idx:];s=s.replace('mv_block->computePredFPel(ref_picture,','mv_block->computePredFPel((++lmp_ie,ref_picture),');s=s.replace('return min_mcost;', '*mv = tmp; return min_mcost;', 2);p.write_text(s)
