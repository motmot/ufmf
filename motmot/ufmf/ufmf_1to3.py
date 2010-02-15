import motmot.ufmf.ufmf as ufmf
import motmot.ufmf.reindex as reindex
from optparse import OptionParser
import tempfile, os, shutil, struct, warnings, stat, sys
import numpy as np

def convert(filename,progress=False,shrink=False, out_fname=None):
    version = ufmf.identify_ufmf_version(filename)

    if version != 1:
        raise ValueError('ufmf_1to3.convert requires a v1 file as input')

    basename = os.path.splitext(filename)[0]
    mean_fmf_filename = basename + '_mean.fmf'
    if not os.path.exists( mean_fmf_filename ):
        raise RuntimeError('conversion from "%s" requires _mean.fmf file "%s", '
                           'but not found' % (filename, mean_fmf_filename) )

    infile_ufmf=ufmf.UfmfV1(filename,
                            use_conventional_named_mean_fmf=True)
    if not infile_ufmf.use_conventional_named_mean_fmf:
        raise RuntimeError('conversion requires _mean.fmf file, but not found')

    in_mean_fmf = infile_ufmf._mean_fmf # naughty access of private variable
    in_sumsqf_fmf = infile_ufmf._sumsqf_fmf # naughty access of private variable
    frame0,timestamp0 = infile_ufmf.get_bg_image()
    max_height, max_width = frame0.shape

    mean_timestamps = infile_ufmf._mean_fmf_timestamps # naughty access of private variable
    diff = mean_timestamps[1:]-mean_timestamps[:-1]
    assert np.all(diff >= 0), 'mean image timestamps not ascending'

    next_mean_ts_idx = 0

    tmp_out_filename = out_fname
    if tmp_out_filename is None:
        tmp_out_filename = filename+'.v3'
        while os.path.exists(tmp_out_filename):
            tmp_out_filename += '.v3'

    if os.path.exists(tmp_out_filename):
        raise RuntimeError('file exists: %s'%tmp_out_filename)

    if progress:
        import progressbar

        widgets=['copying: ', progressbar.Percentage(), ' ',
                 progressbar.Bar(), ' ', progressbar.ETA()]
        pbar=progressbar.ProgressBar(widgets=widgets,
                                     maxval=len(mean_timestamps)).start()

    conversion_success = False
    if shrink:
        cls = ufmf.AutoShrinkUfmfSaverV3
    else:
        cls = ufmf.UfmfSaverV3
    try:
        # open the file with ufmf writer to write header properly
        outfile_ufmf = cls(tmp_out_filename,
                           coding='MONO8',
                           frame0=frame0,
                           timestamp0=timestamp0,
                           max_width=max_width,
                           max_height=max_height,
                           )

        for (timestamp,regions) in infile_ufmf.readframes():
            if (next_mean_ts_idx < len(mean_timestamps) and
                timestamp >= mean_timestamps[next_mean_ts_idx]):
                # add mean/sumsqf image
                meanf, meants = in_mean_fmf.get_frame(next_mean_ts_idx)
                sumsqff, sumsqfts = in_sumsqf_fmf.get_frame(next_mean_ts_idx)
                assert meants==sumsqfts
                assert meants==mean_timestamps[next_mean_ts_idx]
                outfile_ufmf.add_keyframe('mean',meanf,meants)
                outfile_ufmf.add_keyframe('sumsq',sumsqff,sumsqfts)
                next_mean_ts_idx += 1
                if progress:
                    pbar.update(next_mean_ts_idx)
            outfile_ufmf._add_frame_regions(timestamp,regions)

        outfile_ufmf.close()
        conversion_success = True
    finally:
        if not conversion_success:
            if os.path.exists(tmp_out_filename):
                os.unlink(tmp_out_filename)
        if progress:
            pbar.finish()


    assert conversion_success==True
    #backup_orig_fname = filename + '.orig'
    #shutil.move(filename,backup_orig_fname)
    #shutil.move(tmp_out_filename,filename)
    #os.unlink(backup_orig_fname)

def main():
    usage = """%prog FILE [options]

Convert a v1 .ufmf file to v3.

"""

    parser = OptionParser(usage)
    parser.add_option("--shrink", action='store_true', default=False,
                      help="shrink output file")
    parser.add_option("--progress", action='store_true', default=False,
                      help="show a progress bar while indexing file")
    parser.add_option('-o','--outfile', default = None,
                      help="destination file")
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    convert(filename,
            progress=options.progress,
            shrink=options.shrink,
            out_fname=options.outfile,
            )

if __name__=='__main__':
    main()
