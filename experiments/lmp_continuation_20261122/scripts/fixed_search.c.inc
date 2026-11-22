/* Evaluate 18 distinct integer points in the existing JM search bounds.
 * Each center gets the same 3x3 stencil. Fill overlap with points around
 * the native center. JM's fractional refinement still runs afterwards.
 */
static distblk lmp_fixed_search(
    Macroblock *mb, MEBlock *block, StorablePicture *reference,
    MotionVector predictor, MotionVector native, int mode, int lambda,
    MotionVector *best)
{
  MotionVector centers[2] = {native, {0, 0}};
  MotionVector seen[18];
  distblk best_cost = DISTBLK_MAX;
  SearchWindow *range = &block->searchRange;
  int n = 0, j, x, y, k, ix;
  int lo_x = native.mv_x - range->max_x, hi_x = native.mv_x + range->max_x;
  int lo_y = native.mv_y - range->max_y, hi_y = native.mv_y + range->max_y;

  if (mode >= 5)
  {
    if (!lmp_load())
    {
      fprintf(stderr, "LMP: missing or truncated candidate file\n");
      exit(2);
    }
    ix = (((mb->p_Vid->frame_no - 1) * 30 + mb->pix_y / 16) * 52
          + mb->pix_x / 16) * 16;
    centers[1].mv_x = lmp_candidates[ix];
    centers[1].mv_y = lmp_candidates[ix + 1];
    if (mode == 5)
    {
      centers[0] = centers[1];
      centers[1].mv_x = lmp_candidates[ix + 2];
      centers[1].mv_y = lmp_candidates[ix + 3];
    }
  }
  for (j = 0; j < 3 && n < 18; ++j)
  {
    int radius = j == 2 ? 4 : 1;
    MotionVector center = j == 2 ? native : centers[j];
    center.mv_x = (short)iClip3(lo_x, hi_x, center.mv_x);
    center.mv_y = (short)iClip3(lo_y, hi_y, center.mv_y);
    for (y = -radius; y <= radius && n < 18; ++y)
      for (x = -radius; x <= radius && n < 18; ++x)
      {
        MotionVector point = {(short)(center.mv_x + 4*x),
                              (short)(center.mv_y + 4*y)};
        MotionVector padded;
        distblk cost;
        if (point.mv_x < lo_x || point.mv_x > hi_x ||
            point.mv_y < lo_y || point.mv_y > hi_y)
          continue;
        for (k = 0; k < n; ++k)
          if (seen[k].mv_x == point.mv_x && seen[k].mv_y == point.mv_y)
            break;
        if (k != n)
          continue;
        seen[n++] = point;
        padded = pad_MVs(point, block);
        cost = mv_cost(mb->p_Vid, lambda, &padded, &predictor);
        ++lmp_ie;
        cost += block->computePredFPel(reference, block,
                                       DISTBLK_MAX - cost, &padded);
        if (cost < best_cost)
        {
          best_cost = cost;
          *best = point;
        }
      }
  }
  if (n != 18)
  {
    fprintf(stderr, "LMP: fixed search exhausted after %d points\n", n);
    exit(2);
  }
  return best_cost;
}

