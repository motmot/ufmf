import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import sys
import numpy as np
import pprint

def get_timestamps( timestamps, level ):
    if level >= 2:
        min_t = np.min(timestamps)
        max_t = np.max(timestamps)
        return min_t, max_t
    else:
        return None, None

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
    stats = collect_stats(filename,level=options.level)
    pprint.pprint(stats)

def collect_stats(filename,level=3):
    result = {}
    movie = ufmf.Ufmf(filename)
    if not isinstance(movie,ufmf.UfmfV3):
        sys.stderr.write('stats only available for .ufmf movies version 3\n')
        sys.exit(1)
    idx = movie.get_index()
    if level >= 1:
        kf = {}
        result['frame_info'] = kf
    for keyframe_type in idx['keyframe'].keys():
        if level >= 1:
            kf[keyframe_type] = {}
            kf[keyframe_type]['num_frames'] = len(idx['keyframe'][keyframe_type]['loc'])
            if level >= 2:
                timestamps = idx['keyframe'][keyframe_type]['timestamp']
                start, stop = get_timestamps( timestamps, level )
                kf[keyframe_type]['start_time_float'] = start
                kf[keyframe_type]['stop_time_float'] = stop
    if level >= 1:
        keyframe_type = 'raw'
        kf[keyframe_type] = {}
        kf[keyframe_type]['num_frames'] = len(idx['frame']['loc'])
        if level >= 2:
            timestamps = idx['frame']['timestamp']
            start, stop = get_timestamps( timestamps, level )
            kf[keyframe_type]['start_time_float'] = start
            kf[keyframe_type]['stop_time_float'] = stop

        if level >= 3:
            n_frames_with_regions = 0
            min_ts = np.inf
            max_ts = -np.inf
            for (timestamp, regions) in movie.readframes():
                if len(regions):
                    n_frames_with_regions += 1
                    min_ts = min( min_ts, timestamp )
                    max_ts = max( max_ts, timestamp )
            if np.isfinite(min_ts):
                result['data_start_time_float'] = min_ts
                result['data_stop_time_float'] = max_ts
            result['data_num_frames'] = n_frames_with_regions
    return result

