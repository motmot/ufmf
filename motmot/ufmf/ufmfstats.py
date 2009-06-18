import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import sys

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    movie = ufmf.Ufmf(filename)
    idx = movie.get_index()
    for keyframe_type in idx['keyframe'].keys():
        print "keyframe '%s': %d frames"%(
            keyframe_type,
            len(idx['keyframe'][keyframe_type]['loc']))
    print "normal frames: %d frames"%(
        len(idx['frame']['loc']))
