#FlyMovieFormat
from __future__ import division
import sys
import struct
import warnings
import os.path

import numpy
from numpy import nan

import time

import math

# version 1 formats:
HEADER_FMT = '<IIdII'
CHUNKHEADER_FMT = '<dI'
SUBHEADER_FMT = '<II'
TIMESTAMP_FMT = 'd' # XXX struct.pack('<d',nan) dies


class NoMoreFramesException( Exception ):
    pass

class InvalidMovieFileException( Exception ):
    pass

class UfmfParser(object):
    """derive from this class to create your own parser

    you will need to implemented the following functions:

    def handle_bg(self, timestamp0, bg_im):
        pass

    def handle_frame(self, timestamp, regions):
        pass
    """

    def parse(self,filename):
        ufmf = Ufmf(filename)
        bg_im, timestamp0 = ufmf.get_bg_image()

        self.handle_bg(timestamp0, bg_im)

        while 1:
            buf = fd.read( chunkheadsz )
            if not len(buf):
                # no more frames (EOF)
                break
            intup = struct.unpack(CHUNKHEADER_FMT, buf)
            (timestamp, n_pts) = intup
            regions = []
            for ptnum in range(n_pts):
                subbuf = fd.read(subsz)
                intup = struct.unpack(SUBHEADER_FMT, subbuf)
                xmin, ymin = intup

                buf = fd.read( chunkimsize )
                bufim = numpy.fromstring( buf, dtype = numpy.uint8 )
                bufim.shape = chunkheight, chunkwidth
                regions.append( (xmin,ymin, bufim) )

            self.handle_frame(timestamp, regions)
        fd.close()

class Ufmf(object):
    """class to read .ufmf files"""
    bufsz = struct.calcsize(HEADER_FMT)
    chunkheadsz = struct.calcsize( CHUNKHEADER_FMT )
    subsz = struct.calcsize(SUBHEADER_FMT)

    def __init__(self,filename, seek_ok=True):
        mode = "rb"
        self._filename = filename
        self._fd = open(filename,mode=mode)
        buf = self._fd.read( self.bufsz )
        intup = struct.unpack(HEADER_FMT, buf)
        (self._version, self._image_radius,
         self._timestamp0,
         self._width, self._height) = intup
        # extract background
        bg_im_buf = self._fd.read( self._width*self._height)
        self._bg_im = numpy.fromstring( bg_im_buf, dtype=numpy.uint8)
        self._bg_im.shape = self._height, self._width
        if hasattr(self,'handle_bg'):
            self.handle_bg(self._timestamp0, self._bg_im)

        # get ready to extract frames
        self._chunkwidth = 2*self._image_radius
        self._chunkheight = 2*self._image_radius
        self._chunkshape = self._chunkheight, self._chunkwidth
        self._chunkimsize = self._chunkwidth*self._chunkheight
        self._last_safe_x = self._width - self._chunkwidth
        self._last_safe_y = self._height - self._chunkheight
        if seek_ok:
            self._fd_start = self._fd.tell()
            self._fd.seek(0,2)
            self._fd_end = self._fd.tell()
            self._fd.seek(self._fd_start,0)
            self._fd_length = self._fd_end - self._fd_start
        else:
            self._fd_length = None

    def get_bg_image(self):
        """return the background image"""
        return self._bg_im, self._timestamp0

    def get_progress(self):
        """get a fraction of completeness (approximate value)"""
        if self._fd_length is None:
            # seek_ok was false - don't know how long this is
            return 0.0
        dist = self._fd.tell()-self._fd_start
        return dist/self._fd_length

    def readframes(self):
        """return a generator of the frame information"""
        cnt=0
        while 1:
            buf = self._fd.read( self.chunkheadsz )
            cnt+=1
            if not len(buf):
                # no more frames (EOF)
                break
            intup = struct.unpack(CHUNKHEADER_FMT, buf)
            (timestamp, n_pts) = intup

            regions = []
            for ptnum in range(n_pts):
                subbuf = self._fd.read(self.subsz)
                intup = struct.unpack(SUBHEADER_FMT, subbuf)
                xmin, ymin = intup

                if (xmin < self._last_safe_x and
                    ymin < self._last_safe_y):
                    read_length = self._chunkimsize
                    bufshape = self._chunkshape
                else:
                    chunkwidth = min(self._width - xmin, self._chunkwidth)
                    chunkheight = min(self._height - ymin, self._chunkheight)
                    read_length = chunkwidth*chunkheight
                    bufshape = chunkheight,chunkwidth
                buf = self._fd.read( read_length )
                bufim = numpy.fromstring( buf, dtype = numpy.uint8 )
                bufim.shape = bufshape
                regions.append( (xmin,ymin, bufim) )
            yield timestamp, regions

    def close(self):
        self._fd.close()

class UfmfSaver:
    """class to write (save) .ufmf files"""
    def __init__(self,
                 filename,
                 frame0,
                 timestamp0,
                 image_radius=10,
                 ):
        self.filename = filename
        mode = "w+b"
        self.file = open(self.filename,mode=mode)
        self.image_radius = image_radius
        self.version = 1

        bg_frame = numpy.asarray(frame0)
        self.height, self.width = bg_frame.shape
        assert bg_frame.dtype == numpy.uint8
        self.timestamp0 = timestamp0

        self.file.write(struct.pack(HEADER_FMT,
                                    self.version, self.image_radius,
                                    self.timestamp0,
                                    self.width, self.height))
        bg_data = bg_frame.tostring()
        assert len(bg_data)==self.height*self.width
        self.file.write(bg_data)
        self.last_timestamp = self.timestamp0

    def add_frame(self,origframe,timestamp,point_data):
        origframe = numpy.asarray( origframe )

        assert origframe.shape == (self.height, self.width)

        n_pts = len(point_data)
        self.file.write(struct.pack(CHUNKHEADER_FMT, timestamp, n_pts ))
        str_buf = []
        for this_point_data in point_data:
            xidx, yidx = this_point_data[:2]

            xmin = int(round(xidx-self.image_radius))
            xmin = max(0,xmin)

            xmax = xmin + 2*self.image_radius
            xmax = min( xmax, self.width)
            if xmax == self.width:
                xmin = self.width - (2*self.image_radius)

            ymin = int(round(yidx-self.image_radius))
            ymin = max(0,ymin)

            ymax = ymin + 2*self.image_radius
            ymax = min( ymax, self.width)
            if ymax == self.height:
                ymin = self.height - (2*self.image_radius)

            try:
                assert ymax-ymin == (2*self.image_radius)
                assert xmax-xmin == (2*self.image_radius)
            except:
                print 'xmin, xidx, xmax',xmin, xidx, xmax
                print 'ymin, yidx, ymax',ymin, yidx, ymax
                print 'self.image_radius',self.image_radius
                raise

            roi = origframe[ ymin:ymax, xmin:xmax ]
            this_str_buf = roi.tostring()
            this_str_head = struct.pack(SUBHEADER_FMT, xmin, ymin)

            str_buf.append( this_str_head + this_str_buf )
        fullstr = ''.join(str_buf)
        if len(fullstr):
            self.file.write(fullstr)
        self.last_timestamp = timestamp

    def close(self):
        self.file.close()

    def __del__(self):
        if hasattr(self,'file'):
            self.close()
