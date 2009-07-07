import motmot.ufmf.ufmf as ufmf
import motmot.ufmf.reindex as reindex
from optparse import OptionParser
import tempfile, os, shutil, struct, warnings, stat
import numpy as np

def convert(filename,short_file_ok=False,progress=False):
    version = ufmf.identify_ufmf_version(filename)

    if version == 3:
        warnings.warn('ufmf_2to3.convert() called with v3 file as input')
        return

    if version != 2:
        raise ValueError('ufmf_2to3.convert requires a v2 file as input')

    try:
        try:
            # try to use pre-existing index
            infile_ufmf=ufmf.UfmfV2(filename,
                                    short_file_ok=short_file_ok,
                                    index_progress=progress,
                                    )
        except ufmf.CorruptIndexError:
            # regenerate it
            infile_ufmf=ufmf.UfmfV2(filename,
                                    short_file_ok=short_file_ok,
                                    index_progress=progress,
                                    ignore_preexisting_index=True,
                                    is_ok_to_write_regenerated_index=False,
                                    )
    except ufmf.ShortUFMFFileError, err:
        raise ValueError('this file appears to be short. '
                         '(Hint: Retry with the --short-file-ok option.)')

    try:
        frame0,timestamp0 = infile_ufmf.get_bg_image()
    except ufmf.NoMoreFramesException:
        # a background image named 'frame0' does not exist
        frame0,timestamp0 = None, None
    max_width, max_height = infile_ufmf.get_max_size()
    infile_ufmf.close()

    v2_header_size = struct.calcsize(ufmf.FMT[2].HEADER)
    v3_header_size = struct.calcsize(ufmf.FMT[3].HEADER)

    tmp_out_filename = filename+'.v3'
    while os.path.exists(tmp_out_filename):
        tmp_out_filename += '.v3'

    conversion_success = False
    try:
        # open the file with ufmf writer to write header properly
        outfile_ufmf = ufmf.UfmfSaverV3(tmp_out_filename,
                                        coding=infile_ufmf.get_coding(),
                                        frame0=frame0,
                                        timestamp0=timestamp0,
                                        max_width=max_width,
                                        max_height=max_height,
                                        )
        outfile_ufmf.close()

        outfile_fd = open(tmp_out_filename,mode='rb+')

        infile_fd = open(filename,mode='rb')
        infile_fd.seek(v2_header_size)
        outfile_fd.seek(v3_header_size)
        if progress:
            import progressbar

            maxval = os.stat(filename).st_size

            widgets=['copying: ', progressbar.Percentage(), ' ',
                     progressbar.Bar(), ' ', progressbar.ETA()]
            pbar=progressbar.ProgressBar(widgets=widgets,
                                         maxval=maxval).start()
        while 1:
            bytes = infile_fd.read(4096)
            if progress:
                pbar.update(infile_fd.tell())
            if len(bytes)==0:
                break
            outfile_fd.write(bytes)
        infile_fd.close()
        outfile_fd.close()
        if progress:
            pbar.finish()

        reindex.reindex(tmp_out_filename,
                        short_file_ok=short_file_ok,
                        progress=progress)

        conversion_success = True
    finally:
        if not conversion_success:
            if os.path.exists(tmp_out_filename):
                os.unlink(tmp_out_filename)

    assert conversion_success==True
    backup_orig_fname = filename + '.orig'
    shutil.move(filename,backup_orig_fname)
    shutil.move(tmp_out_filename,filename)
    os.unlink(backup_orig_fname)

def main():
    usage = """%prog FILE [options]

Convert a v2 .ufmf file to v3.

"""

    parser = OptionParser(usage)
    parser.add_option("--short-file-ok", action='store_true', default=False,
                      help="don't fail on files that appear to be truncated")
    parser.add_option("--progress", action='store_true', default=False,
                      help="show a progress bar while indexing file")
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    convert(filename,
            short_file_ok=options.short_file_ok,
            progress=options.progress,
            )

def _check_convert_file(is_corrupt):
    orig_fname = reindex._make_temp_ufmf_file(2)
    try:
        converted_fname = orig_fname + '.converted'
        shutil.copyfile(orig_fname,converted_fname)
        try:
            if is_corrupt:
                fd = open(converted_fname,mode='rb+')
                fd.seek(-10, os.SEEK_END)
                fd.truncate()
                fd.close()
            convert(converted_fname)
            # XXX no comparison of frames/index is actually done
        finally:
            os.unlink(converted_fname)
    finally:
        os.unlink(orig_fname)

def test_convert_file():
    for is_corrupt in (True,False):
        yield _check_convert_file, is_corrupt

if __name__=='__main__':
    main()
