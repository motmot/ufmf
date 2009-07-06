import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser

def remove_index(filename,short_file_ok=False):
    version = ufmf.identify_ufmf_version(filename)
    if version == 1:
        raise ValueError('.ufmf v1 files have no index')
    try:
        f=ufmf.UfmfV2(filename,
                      ignore_preexisting_index=True,
                      short_file_ok=short_file_ok,
                      raise_write_errors=True,
                      mode='rb+',
                      )
    except ufmf.ShortUFMFFileError, err:
        raise ValueError('this file appears to be short. '
                         '(Hint: Retry with the --short-file-ok option.)')

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)
    parser.add_option("--short-file-ok", action='store_true', default=False,
                      help="don't fail on files that appear to be truncated")
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    version = ufmf.identify_ufmf_version(filename)
    assert version != 1, 'v1 .ufmf files have no index to remove'
    remove_index(filename,short_file_ok=options.short_file_ok)

