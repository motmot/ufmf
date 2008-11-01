import motmot.FlyMovieFormat.playfmf as playfmf
import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import sys

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)

    parser.add_option("--darken", type='int',default=0,
                      help="show saved regions as darker by this amount")

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]

    if (sys.platform.startswith('win') or
        sys.platform.startswith('darwin')):
        kws = dict(redirect=True,filename='playfmf.log')
    else:
        kws = {}
    app = playfmf.MyApp(**kws)
    flymovie = ufmf.FlyMovieEmulator(filename,darken=options.darken)
    app.OnNewMovie(flymovie)
    app.MainLoop()

if __name__ == '__main__':
    main()
