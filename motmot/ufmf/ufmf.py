from __future__ import division
import sys
import struct
import warnings
import os.path, hashlib

import numpy
import numpy as np
from numpy import nan

import time

import math

import motmot.FlyMovieFormat.FlyMovieFormat as FMF

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
            if len(buf)!=chunkheadsz:
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

    def __init__(self,filename,
                 seek_ok=True,
                 use_conventional_named_mean_fmf=True,
                 ):
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

        self.use_conventional_named_mean_fmf = use_conventional_named_mean_fmf
        if self.use_conventional_named_mean_fmf:
            basename = os.path.splitext(self._filename)[0]
            fmf_filename = basename + '_mean.fmf'
            if os.path.exists(fmf_filename):
                self._mean_fmf = FMF.FlyMovie(fmf_filename)
                self._mean_fmf_timestamps = self._mean_fmf.get_all_timestamps()
                dt=self._mean_fmf_timestamps[1:]-self._mean_fmf_timestamps[:-1]
                assert np.all(dt > 0) # make sure searchsorted will work
            else:
                self.use_conventional_named_mean_fmf = False

    def get_mean_for_timestamp(self, timestamp ):
        if not hasattr(self,'_mean_fmf_timestamps'):
            raise ValueError(
                'ufmf %s does not have mean image data'%self._filename)
        fno=np.searchsorted(self._mean_fmf_timestamps,timestamp,side='right')-1
        mean_image, timestamp_mean = self._mean_fmf.get_frame(fno)
        assert timestamp_mean <= timestamp
        return mean_image

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

    def tell(self):
        return self._fd.tell()

    def seek(self,loc):
        self._fd.seek(loc)

    def readframes(self):
        """return a generator of the frame information"""
        cnt=0
        while 1:
            buf = self._fd.read( self.chunkheadsz )
            cnt+=1
            if len(buf)!=self.chunkheadsz:
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

class NoSuchFrameError(IndexError):
    pass

def md5sum_headtail(filename):
    """quickly calculate a hash value for an even giant file"""
    fd = open(filename,mode='rb')
    start_bytes = fd.read(1000)

    try:
        fd.seek(-1000,os.SEEK_END)
    except IOError,err:
        # it's OK, we'll just read up to another 1000 bytes
        pass

    stop_bytes = fd.read(1000)
    bytes = start_bytes+stop_bytes
    m = hashlib.md5()
    m.update(bytes)
    return m.digest()

class FlyMovieEmulator(object):
    def __init__(self,filename,
                 darken=0,
                 allow_no_such_frame_errors=False,
                 white_background=False,
                 **kwargs):
        self._ufmf = Ufmf(
            filename,**kwargs)
        self._start = self._ufmf.tell()
        self._fno2loc = None
        self._timestamps = None
        self.format = 'MONO8' # by definition
        self._last_frame = None
        self.filename = filename
        self._bg0,self._ts0=self._ufmf.get_bg_image()
        self._darken=darken
        self._allow_no_such_frame_errors = allow_no_such_frame_errors
        if self._ufmf.use_conventional_named_mean_fmf:
            assert white_background==False
        self.white_background = white_background

    def close(self):
        self._ufmf.close()

    def get_n_frames(self):
        self._fill_timestamps_and_locs()
        last_frame = len(self._fno2loc)
        return last_frame+1

    def get_format(self):
        return self.format
    def get_bits_per_pixel(self):
        return 8
    def get_all_timestamps(self):
        self._fill_timestamps_and_locs()
        return self._timestamps
    def get_frame(self,fno,allow_partial_frames=False):
        if allow_partial_frames:
            warnings.warn('unsupported argument "allow_partial_frames" ignored')
        try:
            self.seek(fno)
        except NoSuchFrameError, err:
            if self._allow_no_such_frame_errors:
                raise
            else:
                return self._bg0,self._ts0 # just return first background image
        else:
            return self.get_next_frame()

    def seek(self,fno):
        if 0<= fno < len(self._fno2loc):
            loc = self._fno2loc[fno]
            self._ufmf.seek(loc)
            self._last_frame = None
        else:
            raise NoSuchFrameError('fno %d not in .ufmf file'%fno)

    def get_next_frame(self):
        have_frame = False
        for timestamp, regions in self._ufmf.readframes():
            if self._ufmf.use_conventional_named_mean_fmf:
                self._last_frame = self._ufmf.get_mean_for_timestamp(timestamp)
            elif self.white_background:
                self._last_frame = numpy.empty(self._bg0.shape,dtype=np.uint8)
                self._last_frame.fill(255)
            else:
                if self._last_frame is None:
                    self._last_frame = numpy.array(self._bg0,copy=True)
            have_frame = True
            for xmin,ymin,bufim in regions:
                h,w=bufim.shape
                self._last_frame[ymin:ymin+h, xmin:xmin+w]=\
                                              np.clip(bufim-self._darken, 0,255)
            break # only want 1 frame
        if not have_frame:
            raise NoMoreFramesException('EOF')
        return self._last_frame, timestamp

    def _fill_timestamps_and_locs(self):
        if self._timestamps is not None:
            # already did this
            return

        src_dir, fname = os.path.split(os.path.abspath( self.filename ))
        cache_dir = os.path.join( src_dir, '.ufmf-cache' )
        fname_base = os.path.splitext(fname)[0]
        cache_fname = os.path.join( cache_dir, fname_base+'.cache.npz' )
        try:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)

            my_hash = md5sum_headtail(self.filename)
            assert self._fno2loc is None

            # load results from hash file
            if os.path.exists(cache_fname):
                npz = np.load(cache_fname)
                cache_hash = str(npz['my_hash'])
                if cache_hash==my_hash:
                    self._timestamps = npz['timestamps']
                    self._fno2loc = npz['fno2loc']
                    return
        except Exception, err:
            if int(os.environ.get('UFMF_FORCE_CACHE','0')):
                raise
            else:
                warnings.warn( 'While attempting to open cache in %s: %s '
                               ' (set environment variable '
                               'UFMF_FORCE_CACHE=1 to raise)'%(cache_fname,err))

        # no hash file or stale hash file -- recompute

        self._timestamps = []
        self._fno2loc = []
        start_pos = self._ufmf.tell()
        self._fno2loc.append( start_pos )
        for timestamp, regions in self._ufmf.readframes():
            self._timestamps.append( timestamp )
            self._fno2loc.append( self._ufmf.tell() )
        del self._fno2loc[-1] # remove last entry -- it's at end of file
        self._ufmf.seek( start_pos )

        assert len(self._timestamps)==len(self._fno2loc)

        try:
            # save results to hash file
            timestamps = np.array(self._timestamps)
            fno2loc = np.array(self._fno2loc)
            np.savez(cache_fname,
                     my_hash=my_hash,
                     timestamps=timestamps,
                     fno2loc=fno2loc)
        except Exception,err:
            if int(os.environ.get('UFMF_FORCE_CACHE','0')):
                raise
            else:
                warnings.warn( str(err)+' (set environment variable '
                               'UFMF_FORCE_CACHE=1 to raise)' )

    def get_height(self):
        return self._bg0.shape[0]
    def get_width(self):
        return self._bg0.shape[1]

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

            assert ymax-ymin == (2*self.image_radius)
            assert xmax-xmin == (2*self.image_radius)

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
