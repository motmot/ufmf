import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import tempfile, os, shutil
import numpy as np

def reindex(filename,short_file_ok=False,progress=False):
    version = ufmf.identify_ufmf_version(filename)
    if version == 1:
        raise ValueError('.ufmf v1 files have no index')
    elif version in (2,3):
        try:
            if version == 2:
                cls = ufmf.UfmfV2
            elif version==3:
                cls = ufmf.UfmfV3

            f=cls(filename,
                          ignore_preexisting_index=True,
                          short_file_ok=short_file_ok,
                          raise_write_errors=True,
                          mode='rb+',
                          index_progress=progress,
                          )
        except ufmf.ShortUFMFFileError, err:
            raise ValueError('this file appears to be short. '
                             '(Hint: Retry with the --short-file-ok option.)')
    else:
        raise ValueError('unknown .ufmf file version')

def main():
    usage = """%prog FILE [options]

Re-index the .ufmf file named FILE and save the index into the FILE.

This can be used to fix corrupt .ufmf indexes.
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
    version = ufmf.identify_ufmf_version(filename)
    assert version != 1, 'v1 .ufmf files have no index to remove'
    reindex(filename,
            short_file_ok=options.short_file_ok,
            progress=options.progress,
            )

def _make_temp_ufmf_file(version):
    tmp_fd,filename = tempfile.mkstemp()
    os.fdopen(tmp_fd).close()

    w = 752
    h = 240

    frame1 = np.zeros( (h,w), dtype = np.uint8 )
    frame1[20:30,30:32] = 255
    frame1[20:30,30:32] = 255

    timestamp = 1.0
    radius = 15

    kwargs = dict(max_width=w,
                  max_height=h,
                  )

    subw = 2*radius
    subh = 2*radius
    us = ufmf.UfmfSaver( filename,
                         frame1,
                         timestamp,
                         version=version,
                         **kwargs)
    assert isinstance(us,ufmf.UfmfSaverBase)
    frame2 = np.zeros( (h,w), dtype = np.uint8 )
    frame2[:,0::2] = range(0, w, 2) # clips (broadcast)
    for i in range(h):
        frame2[i,1::2] = i

    ll_pts = [
        [ (345, 144),
          (521, 118),
          ],
        [ (347, 144),
          (522, 119),
          (367, 229),
          ],
        [(349,145),
         (522,120),
         (369,229),
         ],
        ]
    all_pts = [ [(x+radius,y+radius,subw,subh) for (x,y) in pts ] for pts in ll_pts ]
    for pts in all_pts:
        timestamp += 1.0
        us.add_frame( frame2, timestamp, pts )
    us.close()
    return filename

def _check_reindex_file(is_corrupt,version):
    orig_fname = _make_temp_ufmf_file(version)
    try:
        reindexed_fname = orig_fname + '.reindexed'
        shutil.copyfile(orig_fname,reindexed_fname)
        try:
            if is_corrupt:
                fd = open(reindexed_fname,mode='rb+')
                fd.seek(-10, os.SEEK_END)
                fd.truncate()
                fd.close()
            reindex(reindexed_fname)
            expected = open(orig_fname,'rb').read()
            actual = open(reindexed_fname,'rb').read()
            assert actual==expected
        finally:
            os.unlink(reindexed_fname)
    finally:
        os.unlink(orig_fname)

def test_reindex_file():
    for is_corrupt in (True,False):
        for version in (2,3):
            yield _check_reindex_file, is_corrupt, version

if __name__=='__main__':
    main()
