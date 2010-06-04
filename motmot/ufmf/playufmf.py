import motmot.FlyMovieFormat.playfmf as playfmf
import motmot.ufmf.ufmf as ufmf
from optparse import OptionParser
import sys

def main():
    usage = '%prog FILE [options]'

    parser = OptionParser(usage)

    parser.add_option("--darken", type='int',default=0,
                      help="show saved regions as darker by this amount")

    parser.add_option("--white-background", action='store_true', default=False,
                      help="don't display background information")

    parser.add_option("--force-no-mean-fmf", action='store_false', default=True,
                      help="disable use of FILE_mean.fmf as background image source")

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.print_help()
        return

    filename = args[0]
    version = ufmf.identify_ufmf_version(filename)

    if (sys.platform.startswith('win') or
        sys.platform.startswith('darwin')):
        kws = dict(redirect=True,filename='playufmf.log')
    else:
        kws = {}
    app = playfmf.MyApp(**kws)

    kwargs = dict(white_background=options.white_background,
                  )
    if version==1:
        use_fmf = options.force_no_mean_fmf
        if options.white_background:
            use_fmf = False
        kwargs.update(dict(use_conventional_named_mean_fmf=use_fmf,
                           ))

    flymovie = ufmf.FlyMovieEmulator(filename,
                                     darken=options.darken,
                                     **kwargs)
    app.OnNewMovie(flymovie)
    app.MainLoop()

if __name__ == '__main__':
    main()
