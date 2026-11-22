from pathlib import Path
import shutil,re,difflib,json
R=Path.home()/'lmp-e1-e2-followup-20261121';D=R/'jm-perblock';O=R/'supplement-analysis'
shutil.copytree(R/'jm-d3',D,dirs_exist_ok=True)
p=D/'lencod/src/me_epzs_int.c';s=p.read_text();globals='''
int lmp_seed_evals=0,lmp_seed_improved=0,lmp_seed_duplicate=0,lmp_seed_status=0,lmp_raised=0,lmp_predictors=0,lmp_pattern=0,lmp_reason=0;
double lmp_initial_cost=0,lmp_seed_cost=-1,lmp_after_seed=0,lmp_initial_stop=0;
''';s=globals+s
start=s.index('EPZS_integer_motion_estimation (');end=s.index('\n/*!',start);func=s[start:end]
marker='  Slice *currSlice = currMB->p_Slice;';func=func.replace(marker,'''  lmp_seed_evals=lmp_seed_improved=lmp_seed_duplicate=lmp_seed_status=lmp_raised=lmp_predictors=lmp_pattern=lmp_reason=0;
  lmp_initial_cost=lmp_after_seed=lmp_initial_stop=0;lmp_seed_cost=-1;
'''+marker,1)
marker='  /* Cached latent proposal;';func=func.replace(marker,'  lmp_initial_cost=(double)min_mcost;lmp_initial_stop=(double)stopCriterion;\n'+marker,1)
func=func.replace('if(mode>=2 && lmp_load())','if(mode>=2 && !lmp_load())lmp_seed_status=1;\n    if(mode>=2 && lmp_load())',1)
func=func.replace('      if (iabs(proposal.mv_x', '      lmp_seed_status=2;\n      if (iabs(proposal.mv_x',1)
func=func.replace('        MotionVector padded=pad_MVs(proposal,mv_block);','        MotionVector padded=pad_MVs(proposal,mv_block);\n        lmp_seed_duplicate=(proposal.mv_x==mv->mv_x && proposal.mv_y==mv->mv_y);',1)
func=func.replace('((++lmp_ie,ref_picture),','((++lmp_ie,++lmp_seed_evals,ref_picture),',1)
func=func.replace('used=1;','used=1;lmp_seed_cost=(double)cost;lmp_seed_status=3;',1)
func=func.replace('improved=1;}','improved=1;lmp_seed_status=4;}',1)
func=func.replace('if(mode==1 || (mode==3 && improved))stopCriterion*=2;','if(mode==1 || (mode==3 && improved)){stopCriterion*=2;lmp_raised=1;if(mode==3)lmp_seed_status=5;}\n    lmp_seed_improved=improved;lmp_after_seed=(double)min_mcost;',1)
func=re.sub(r'    fprintf\(stderr,"LMP_BLOCK.*?;', '',func,count=1)
func=func.replace('  if (min_mcost > stopCriterion)\n  {','  if (min_mcost > stopCriterion)\n  {\n    lmp_predictors=1;',1)
func=func.replace('    //! Check all predictors','    //! Check all predictors',1)
# Record entering the pattern traversal, not individual point clocks.
func=func.replace('    if (min_mcost > stopCriterion)\n    {','    if (min_mcost > stopCriterion)\n    {\n      lmp_pattern=1;',1)
reasons=['7','2','3','4','(lmp_pattern?6:(lmp_predictors?5:1))'];parts=func.split('return min_mcost;');assert len(parts)==6
func=parts[0]+''.join('lmp_reason='+reason+';return min_mcost;'+part for reason,part in zip(reasons,parts[1:]));s=s[:start]+func+s[end:];p.write_text(s)
p=D/'lencod/src/mv_search.c';s=p.read_text();header='''
extern int lmp_seed_evals,lmp_seed_improved,lmp_seed_duplicate,lmp_seed_status,lmp_raised,lmp_predictors,lmp_pattern,lmp_reason;
extern double lmp_initial_cost,lmp_seed_cost,lmp_after_seed,lmp_initial_stop;
static FILE *lmp_block_file=NULL;
static void lmp_close_blocks(void){if(lmp_block_file)fclose(lmp_block_file);}
static void lmp_open_blocks(void){char *p=getenv("LMP_BLOCK_CSV");if(p && !lmp_block_file){lmp_block_file=fopen(p,"w");if(!lmp_block_file){perror("LMP_BLOCK_CSV");exit(2);}atexit(lmp_close_blocks);fprintf(lmp_block_file,"frame,x,y,blocktype,list,ref,integer_evals,subpixel_evals,seed_evals,seed_duplicate_initial,seed_improved,seed_status,threshold_raised,predictors_entered,pattern_entered,exit_reason,initial_cost,seed_cost,after_seed_cost,initial_stop,me_mv_x,me_mv_y\\n");}}
''';marker='static int lmp_ready=0;';s=s.replace(marker,marker+'\n'+header,1)
marker='  //--- perform motion search ---';s=s.replace(marker,'''  { unsigned long long lmp_i0=lmp_ie,lmp_s0=lmp_se;
'''+marker,1)
marker='  // clip mvs after me is performed';s=s.replace(marker,'''  if(blocktype==1 && list==0 && ref==0){lmp_open_blocks();if(lmp_block_file)fprintf(lmp_block_file,"%d,%d,%d,%d,%d,%d,%llu,%llu,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.0f,%.0f,%.0f,%d,%d\\n",p_Vid->frame_no,currMB->pix_x,currMB->pix_y,blocktype,list,ref,lmp_ie-lmp_i0,lmp_se-lmp_s0,lmp_seed_evals,lmp_seed_duplicate,lmp_seed_improved,lmp_seed_status,lmp_raised,lmp_predictors,lmp_pattern,lmp_reason,lmp_initial_cost,lmp_seed_cost,lmp_after_seed,lmp_initial_stop,mv->mv_x,mv->mv_y);}
  }
'''+marker,1);p.write_text(s)
patch=[]
for name in ['me_epzs_int.c','mv_search.c']:
 a=R/'jm-d3/lencod/src'/name;b=D/'lencod/src'/name;patch.extend(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile='a/lencod/src/'+name,tofile='b/lencod/src/'+name))
(O/'perblock_instrumentation.patch').write_text(''.join(patch))
(O/'reason_codes.json').write_text(json.dumps(dict(exit_reason={'1':'initial_threshold_stop','2':'derived_threshold_stop','3':'reference_prune_after_predictors','4':'reference_prune_in_pattern','5':'predictor_threshold_stop','6':'pattern_completed','7':'reference_prune_before_predictors'},seed_status={'0':'native_only_or_disabled','1':'candidate_file_unavailable','2':'candidate_out_of_search_range','3':'seed_not_better_native_search_retained','4':'seed_better_native_threshold_retained','5':'seed_better_relaxed_threshold_requested'},frozen_policy='unchanged from prior phase; no new gates tuned'),indent=2))
