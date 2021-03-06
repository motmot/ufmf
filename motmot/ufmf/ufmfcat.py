from optparse import OptionParser
import sys
import motmot.ufmf.ufmf as FMF
import numpy

if 1:
    import signal

    # http://mail.python.org/pipermail/python-list/2004-June/268512.html
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def doit(
    filename,
    raten=25,  # numerator
    rated=1,  # denom
    aspectn=1,  # numerator
    aspectd=1,  # denom
    rotate_180=False,
):
    fmf = FMF.FlyMovieEmulator(filename)
    if fmf.get_format() not in ["MONO8", "RAW8"]:
        raise NotImplementedError(
            "Only MONO8 and RAW8 formats are currently supported."
        )
    width = fmf.get_width() // (fmf.get_bits_per_pixel() // 8)
    height = fmf.get_height()

    Y4M_MAGIC = "YUV4MPEG2"
    Y4M_FRAME_MAGIC = "FRAME"

    inter = "Ip"  # progressive
    colorspace = "Cmono"

    out_fd = sys.stdout

    out_fd.write(
        "%(Y4M_MAGIC)s W%(width)d H%(height)d F%(raten)d:%(rated)d %(inter)s A%(aspectn)d:%(aspectd)d %(colorspace)s\n"
        % locals()
    )
    while 1:
        try:
            frame, timestamp = fmf.get_next_frame()
        except FMF.NoMoreFramesException as err:
            break

        out_fd.write("%(Y4M_FRAME_MAGIC)s\n" % locals())

        if rotate_180:
            frame = numpy.rot90(numpy.rot90(frame))

        for i in range(height):
            out_fd.write(frame[i, :].tostring())
        out_fd.flush()


def main():
    usage = """%prog FILENAME [options]

Pipe the contents of an .ufmf file to stdout in the yuv4mpegpipe
format. This allows an .ufmf file to be converted to any format that
ffmpeg supports. For example, to convert the file x.ufmf to x.avi using
the MPEG4 codec:

%prog x.ufmf | ffmpeg -vcodec msmpeg4v2 -i - x.avi

Note that due to issues I don't understand, the following bash script
might be useful:

FAIL=1

while [ $FAIL -ne 0 ]
do
    sleep 1
    echo "trying"
    ufmfcat $1 --rotate-180 | ffmpeg -vcodec msmpeg4v2 -i - -y trash.avi
    FAIL=$?
    echo "FAIL=$FAIL"
done
echo "done"

"""

    parser = OptionParser(usage)

    parser.add_option("--rotate-180", action="store_true", default=False)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        return

    filename = args[0]

    doit(
        filename=args[0], rotate_180=options.rotate_180,
    )


if __name__ == "__main__":
    main()
