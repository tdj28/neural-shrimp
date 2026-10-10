"""Resource profile only: exact N12 refinement horizon on unrelated matrix."""
import sys
import refine_clark_rnn_n12_p6 as R
if __name__=='__main__':
    if '--mode' not in sys.argv or sys.argv[sys.argv.index('--mode')+1]!='profile':raise ValueError('profile-only adapter')
    R.M.BURN=24000;R.M.ALIGN=2000;R.M.TOTAL=34192;R.M.EXTRA=R.EXTRA
    raise SystemExit(R.M.main())
