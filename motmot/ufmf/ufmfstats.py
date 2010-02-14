import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import sys
import numpy as np

def disp_timestamps( timestamps, level ):
    if level >= 2:
        min_t = np.min(timestamps)
        max_t = np.max(timestamps)
        print '  timestamp range: %s - %s'%(min_t, max_t)
        if level >= 10:
            print '  timestamps:',timestamps

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)
    parser.add_option("--level", type='int', default=3,
                      help="level of information to display")
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    movie = ufmf.Ufmf(filename)
    if not isinstance(movie,ufmf.UfmfV3):
        sys.stderr.write('stats only available for .ufmf movies version 3\n')
        sys.exit(1)
    idx = movie.get_index()
    for keyframe_type in idx['keyframe'].keys():
        if options.level >= 1:
            print "keyframe '%s': %d frames"%(
                keyframe_type,
                len(idx['keyframe'][keyframe_type]['loc']))
            if options.level >= 2:
                timestamps = idx['keyframe'][keyframe_type]['timestamp']
                disp_timestamps( timestamps, options.level )
    if options.level >= 1:
        print "normal frames: %d frames"%(
            len(idx['frame']['loc']))
        timestamps = idx['frame']['timestamp']
        disp_timestamps( timestamps, options.level )
        if options.level >= 3:
            n_frames_with_regions = 0
            min_ts = np.inf
            max_ts = -np.inf
            for (timestamp, regions) in movie.readframes():
                if len(regions):
                    n_frames_with_regions += 1
                    min_ts = min( min_ts, timestamp )
                    max_ts = max( max_ts, timestamp )
            if np.isfinite(min_ts):
                tstr = ' (timestamp range: %s - %s)'%(min_ts,max_ts)
            else:
                tstr = ''
            print '  n frames with data: %d%s'%(n_frames_with_regions,tstr)

