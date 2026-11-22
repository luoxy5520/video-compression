from pathlib import Path
import shutil
root=Path.cwd()/'jm-trace'
src=Path.home()/'a_new48_20261107/kit/linux/JM'
shutil.copytree(src,root,ignore=shutil.ignore_patterns('obj','*.o','*.exe'))
p=root/'lencod/src/mv_search.c'
s=p.read_text()
marker='  //=====   INTEGER-PEL SEARCH   ====='
assert s.count(marker)==1
offset=s.rindex('  //==================================',0,s.index(marker))
trace='''  if (blocktype == 1 && list == 0 && ref == 0)
    fprintf(stderr, "LMP_MVP %d %d %d %d %d\\n", p_Vid->frame_no,
            currMB->pix_x, currMB->pix_y, pred.mv_x, pred.mv_y);

'''
p.write_text(s[:offset]+trace+s[offset:])
