import sys, os
import motmot.ufmf.ufmf as ufmf
import motmot.FlyMovieFormat.FlyMovieFormat as FlyMovieFormat

def ufmf2fmf( fname_in, fname_out ):
    u = ufmf.FlyMovieEmulator(fname_in)
    ts = u.get_all_timestamps()
    f = FlyMovieFormat.FlyMovieSaver(fname_out)
    for i in range(len(ts)):
        print 'frame %d of %d'%(i,len(ts))
        frame,timestamp =u.get_next_frame()
        f.add_frame( frame, timestamp )
    f.close()

def main():
    fname_in = sys.argv[1]
    assert fname_in.endswith('.ufmf')
    basefname = os.path.splitext(fname_in)[0]
    fname_out = basefname +'.fmf'
    ufmf2fmf(fname_in, fname_out)

if __name__=='__main__':
    main()

