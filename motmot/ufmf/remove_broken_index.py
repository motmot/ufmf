import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser

def remove_index(filename):
    print 'filename',filename
    try:
        f=ufmf.Ufmf(filename)
    except ufmf.PreexistingIndexExists, index_err:
        fd = open(filename,mode='rb+')
        fd.truncate( index_err.loc )

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)
    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    version = ufmf.identify_ufmf_version(filename)
    assert version != 1, 'v1 .ufmf files have no index to remove'
    remove_index(filename)
    
