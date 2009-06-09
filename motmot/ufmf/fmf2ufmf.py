from __future__ import with_statement
import numpy as np
from optparse import OptionParser
import sys, os
import motmot.FlyMovieFormat.FlyMovieFormat as fmf_mod
import motmot.ufmf.ufmf as ufmf_mod
import contextlib

def doit(filename=None, output_filename=None, hide_gui=False):
    fmf = fmf_mod.FlyMovie(filename)
    ufmf_writer = ufmf_mod.UfmfSaverV2(output_filename,
                                       max_width=fmf.get_width(),
                                       max_height=fmf.get_height(),
                                       coding=fmf.get_format())
    fake_point = ( fmf.get_width()//2, fmf.get_height()//2,
                   fmf.get_width(), fmf.get_height())
    points = [fake_point]
    with contextlib.closing(ufmf_writer):
        try:
            while 1:
                frame, timestamp = fmf.get_next_frame()
                ufmf_writer.add_frame( frame, timestamp, points)
        except fmf_mod.NoMoreFramesException, err:
            pass

def main():
    usage = """%prog FILENAME [options]"""
    parser = OptionParser(usage)

    parser.add_option('--hide-gui', action='store_true',
                      default=False )

    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.print_help()
        sys.exit(1)

    filename = args[0]
    base,ext = os.path.splitext(filename)
    if ext =='.fmf':
        output_filename = base + '.ufmf'
    else:
        output_filename = filename + '.ufmf'
    doit( filename = filename,
          output_filename = output_filename,
          hide_gui = options.hide_gui,
          )

if __name__=='__main__':
    main()
