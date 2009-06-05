from __future__ import division
import sys
import struct, collections
import warnings
import os.path, hashlib
import os, stat

import numpy
import numpy as np
from numpy import nan

import time

import math

import motmot.FlyMovieFormat.FlyMovieFormat as FMF

class BaseDict(dict):
    def __getattr__(self,name):
        return self[name]
    def __setattr__(self,name,value):
        self[name]=value

VERSION_FMT = '<I' # always the first bytes
FMT = {1:BaseDict(HEADER = '<IIdII', # version, ....
                  CHUNKHEADER = '<dI',
                  SUBHEADER = '<II',
                  TIMESTAMP = 'd', # XXX struct.pack('<d',nan) dies
                  ),
       2:BaseDict(HEADER = '<IB', # version, raw coding string length
                  CHUNKID = '<B', # 0 = keyframe, 1 = points
                  KEYFRAME1 = '<B', # (type name)
                  KEYFRAME2 = '<cIId', # (dtype, width,height,timestamp)
                  POINTS1 = '<dI', # timestamp, n_pts
                  POINTS2 = '<HHHH', # x0, y0, w, h
                  ),
       }
KEYFRAME_CHUNK = 0
FRAME_CHUNK = 1

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
            intup = struct.unpack(FMT[1].CHUNKHEADER, buf)
            (timestamp, n_pts) = intup
            regions = []
            for ptnum in range(n_pts):
                subbuf = fd.read(subsz)
                intup = struct.unpack(FMT[1].SUBHEADER, subbuf)
                xmin, ymin = intup

                buf = fd.read( chunkimsize )
                bufim = numpy.fromstring( buf, dtype = numpy.uint8 )
                bufim.shape = chunkheight, chunkwidth
                regions.append( (xmin,ymin, bufim) )

            self.handle_frame(timestamp, regions)
        fd.close()

def identify_ufmf_version(filename):
    mode = "rb"
    fd = open(filename,mode=mode)
    version_buflen = struct.calcsize(VERSION_FMT)
    version_buf = fd.read( version_buflen )
    version, = struct.unpack(VERSION_FMT, version_buf)
    fd.close()
    return version

def Ufmf(filename,**kwargs):
    """factory function to return UfmfBase class instance"""
    version = identify_ufmf_version(filename)
    if version==1:
        return UfmfV1(filename,**kwargs)
    elif version==2:
        if 'seek_ok' in kwargs:
            kwargs.pop('seek_ok')
        return UfmfV2(filename,**kwargs)
    else:
        raise ValueError('unknown .ufmf version %d'%version)

class UfmfBase(object):
    pass

class UfmfV1(UfmfBase):
    """class to read .ufmf version 1 files"""
    bufsz = struct.calcsize(FMT[1].HEADER)
    chunkheadsz = struct.calcsize( FMT[1].CHUNKHEADER )
    subsz = struct.calcsize(FMT[1].SUBHEADER)

    def __init__(self,filename,
                 seek_ok=True,
                 use_conventional_named_mean_fmf=True,
                 ):
        super(UfmfV1,self).__init__()
        mode = "rb"
        self._filename = filename
        self._fd = open(filename,mode=mode)
        buf = self._fd.read( self.bufsz )
        intup = struct.unpack(FMT[1].HEADER, buf)
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
        self._sumsqf_fmf = None
        if self.use_conventional_named_mean_fmf:
            basename = os.path.splitext(self._filename)[0]
            fmf_filename = basename + '_mean.fmf'
            if os.path.exists(fmf_filename):
                self._mean_fmf = FMF.FlyMovie(fmf_filename)
                self._mean_fmf_timestamps = self._mean_fmf.get_all_timestamps()
                dt=self._mean_fmf_timestamps[1:]-self._mean_fmf_timestamps[:-1]
                assert np.all(dt > 0) # make sure searchsorted will work

                sumsqf_filename = basename + '_sumsqf.fmf'
                if os.path.exists(sumsqf_filename):
                    self._sumsqf_fmf = FMF.FlyMovie(sumsqf_filename)
            else:
                self.use_conventional_named_mean_fmf = False

    def get_mean_for_timestamp(self, timestamp, _return_more=False ):
        if not hasattr(self,'_mean_fmf_timestamps'):
            raise ValueError(
                'ufmf %s does not have mean image data'%self._filename)
        fno=np.searchsorted(self._mean_fmf_timestamps,timestamp,side='right')-1
        mean_image, timestamp_mean = self._mean_fmf.get_frame(fno)
        assert timestamp_mean <= timestamp
        if _return_more:
            # assume same times as mean image
            sumsqf_image, timestamp_sumsqf = self._sumsqf_fmf.get_frame(fno)
            return mean_image, sumsqf_image
        else:
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
        while 1:
            buf = self._fd.read( self.chunkheadsz )
            if len(buf)!=self.chunkheadsz:
                # no more frames (EOF)
                break
            intup = struct.unpack(FMT[1].CHUNKHEADER, buf)
            (timestamp, n_pts) = intup

            regions = []
            for ptnum in range(n_pts):
                subbuf = self._fd.read(self.subsz)
                intup = struct.unpack(FMT[1].SUBHEADER, subbuf)
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

class UfmfV2(UfmfBase):
    """class to read .ufmf version 2 files"""

    def __init__(self,file):
        super(UfmfV2,self).__init__()
        self._fd_length = None
        if hasattr(file,'read'):
            # file-like object
            self._file_opened=False
            self._fd = file
        else:
            # filename
            stat_result = os.stat(file)
            self._fd_length = stat_result[stat.ST_SIZE]
            self._fd = open(file,mode='rb')
            self._file_opened=True
        self._fd_start = self._fd.tell()

        bufsz = struct.calcsize(FMT[2].HEADER)
        buf = self._fd_read( bufsz )
        intup = struct.unpack(FMT[2].HEADER, buf)
        (self._version, coding_str_len) = intup
        self._coding = self._fd_read( coding_str_len )

        # index of start pos of each frame chunk
        self._frame_locations = []
        # N of most recently read frame
        self._cur_frame_idx = -1

        # index of start pos of each keyframe chunk (by keyframe_type)
        self._keyframe_locations = collections.defaultdict(list)

        # N of most recently read keyframe (by keyframe_type)
        self._cur_keyframe_idx = {}

        if self._fd_length is None:
            fd_here = self._fd.tell()
            self._fd.seek(0,2)
            fd_end = self._fd.tell()
            self._fd.seek(fd_here,0)
            self._fd_length = fd_end - self._fd_start
        self._keyframe2_sz = struct.calcsize(FMT[2].KEYFRAME2)
        self._points1_sz = struct.calcsize(FMT[2].POINTS1)
        self._points2_sz = struct.calcsize(FMT[2].POINTS2)

    def get_progress(self):
        """get a fraction of completeness (approximate value)"""
        if self._fd_length is None:
            # seek_ok was false - don't know how long this is
            return 0.0
        dist = self._fd.tell()-self._fd_start
        return dist/self._fd_length

    def _get_keyframe_N(self, keyframe_type, N):
        """get Nth keyframe of type keyframe_type"""
        if keyframe_type in self._keyframe_locations:
            self._keyframe_locations[keyframe_type]=[]
        locations = self._keyframe_locations[keyframe_type]
        assert N >= 0
        result = None
        while N >= len(locations):
            # location not yet known
            chunk_id, result = self._index_next_chunk()
            if chunk_id is None:
                raise ValueError('sought keyframe that is not in file')
        if result is None:
            # We didn't need to entire while loop -- we know frame location.
            location = locations[N]
            self._seek(location+1)
            result = self._read_keyframe_chunk(location,return_frame=True)
        else:
            # If we did while loop (above), we stopped on good result.
            assert chunk_id == KEYFRAME_CHUNK
        test_keyframe_type,frame,timestamp=result
        assert keyframe_type==test_keyframe_type
        return frame,timestamp

    def _seek(self,loc):
        self._fd.seek(loc,0)

    def _read_keyframe_chunk(self,start_location):
        """read keyframe chunk from just after chunk_id byte

        start_location is the file locate at which the chunk started.
        (i.e. curpos-1)
        """
        len_type = ord(self._fd_read(1))
        keyframe_type = self._fd_read(len_type)
        assert len(keyframe_type)==len_type
        intup = struct.unpack(FMT[2].KEYFRAME2,
                              self._fd_read(self._keyframe2_sz))
        dtype_char,width,height,timestamp=intup

        previous_idx = self._cur_keyframe_idx.get(keyframe_type,-1)
        this_idx = previous_idx + 1
        locations = self._keyframe_locations[keyframe_type]
        if this_idx >= len(locations):
            assert this_idx==len(locations)
            locations.append(start_location)
        self._cur_keyframe_idx[keyframe_type] = this_idx

        if dtype_char=='B':
            dtype=np.uint8
            sz=1
        elif dtype_char=='f':
            dtype=np.float32
            sz=4
        else:
            pos = self._fd.tell()
            raise ValueError('unknown dtype char')
        read_len = width*height*sz
        buf = self._fd_read(read_len)
        frame = np.fromstring(buf,dtype=dtype)
        frame.shape = (height,width)
        return keyframe_type,frame,timestamp

    def _read_frame_chunk(self,start_location):
        """read frame chunk from just after chunk_id byte

        start_location is the file locate at which the chunk started
        (i.e. curpos-1)
        """
        intup = struct.unpack(FMT[2].POINTS1,
                              self._fd_read(self._points1_sz))
        timestamp, n_pts = intup
        self._cur_frame_idx += 1
        if self._cur_frame_idx >= len(self._frame_locations):
            assert self._cur_frame_idx == len(self._frame_locations)
            self._frame_locations.append(start_location)
        regions = []
        for ptno in range(n_pts):
            intup = struct.unpack(FMT[2].POINTS2,
                                  self._fd_read(self._points2_sz))
            (xmin, ymin, w, h) = intup
            lenbuf = w*h
            buf = self._fd_read(lenbuf)
            im = np.fromstring(buf,dtype=np.uint8)
            im.shape = (h,w)
            regions.append( (xmin,ymin,im) )
        return timestamp,regions

    def readframes(self):
        """return a generator of the frame information"""
        while 1:
            chunk_id, result = self._index_next_chunk()
            if chunk_id==FRAME_CHUNK:
                yield result # (timestamp,regions)
            elif chunk_id is None:
                break # no more frames

    def _fd_read(self,n_bytes,short_OK=False):
        buf = self._fd.read(n_bytes)
        if len(buf)!=n_bytes:
            if not short_OK:
                raise ValueError('expected %d bytes, got %d: short file?'%(
                    n_bytes,len(buf)))
        return buf

    def _index_next_chunk(self):
        loc = self._fd.tell()
        chunk_id_str = self._fd_read(1,short_OK=True)
        if chunk_id_str == '':
            return None, None
        chunk_id = ord(chunk_id_str)
        if chunk_id==KEYFRAME_CHUNK:
            result = self._read_keyframe_chunk(loc)
        elif chunk_id==FRAME_CHUNK:
            # read frame chunk
            result = self._read_frame_chunk(loc)
        else:
            raise ValueError('unexpected byte where chunk ID expected')
        return chunk_id, result

    def get_bg_image(self):
        """return the first raw image (for compatability with UfmfV1)"""
        return self._get_keyframe_N('frame0',0)

    def close(self):
        if self._file_opened:
            self._fd.close()
            self._file_opened = False

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
                 abs_diff=False,
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
        self.abs_diff = abs_diff
        if self.abs_diff:
            assert self._ufmf.use_conventional_named_mean_fmf

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
    def get_frame(self,fno,allow_partial_frames=False,_return_more=False):
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
            return self.get_next_frame(_return_more=_return_more)

    def seek(self,fno):
        if 0<= fno < len(self._fno2loc):
            loc = self._fno2loc[fno]
            self._ufmf.seek(loc)
            self._last_frame = None
        else:
            raise NoSuchFrameError('fno %d not in .ufmf file'%fno)

    def get_next_frame(self, _return_more=False):
        have_frame = False
        more = {}
        for timestamp, regions in self._ufmf.readframes():
            if self._ufmf.use_conventional_named_mean_fmf:
                tmp=self._ufmf.get_mean_for_timestamp(timestamp,
                                                      _return_more=_return_more)
                if _return_more:
                    mean_image, sumsqf_image = tmp
                    more['sumsqf'] = sumsqf_image
                else:
                    mean_image = tmp
                self._last_frame = np.array(mean_image,copy=True).astype(np.uint8)
                more['mean'] = mean_image
            elif self.white_background:
                self._last_frame = numpy.empty(self._bg0.shape,dtype=np.uint8)
                self._last_frame.fill(255)
            else:
                if self._last_frame is None:
                    self._last_frame = numpy.array(self._bg0,copy=True)
            have_frame = True
            more['regions'] = regions
            for xmin,ymin,bufim in regions:
                h,w=bufim.shape
                self._last_frame[ymin:ymin+h, xmin:xmin+w]=\
                                              np.clip(bufim-self._darken, 0,255)
            if self.abs_diff:
                self._last_frame=abs(self._last_frame.astype(np.float32)-
                                     mean_image.astype(np.float32))
                self._last_frame = np.clip(self._last_frame,0,255).astype(np.uint8)
            break # only want 1 frame
        if not have_frame:
            raise NoMoreFramesException('EOF')
        if _return_more:
            return self._last_frame, timestamp, more
        else:
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

def UfmfSaver( file,
               frame0=None,
               timestamp0=None,
               **kwargs):
    """factory function to return UfmfSaverBase instance"""
    default_version = 2
    version = kwargs.pop('version',default_version)
    if version is None:
        version = default_version

    if version==1:
        return UfmfSaverV1(file,frame0,timestamp0,**kwargs)
    elif version==2:
        us = UfmfSaverV2(file,frame0=frame0,timestamp0=timestamp0,**kwargs)
        if frame0 is not None:
            # the frame0, timestamp0 kwargs are cruft from v1 files
            us.add_keyframe('mean',frame0,timestamp0)
        return us
    else:
        raise ValueError('unknown version %s'%version)

class UfmfSaverBase(object):
    def __init__(self,version):
        self.version = version

class UfmfSaverV1(UfmfSaverBase):
    """class to write (save) .ufmf v1 files"""
    def __init__(self,
                 filename,
                 frame0,
                 timestamp0,
                 image_radius=10,
                 ):
        super(UfmfSaverV1,self).__init__(1)
        self.filename = filename
        mode = "w+b"
        self.file = open(self.filename,mode=mode)
        self.image_radius = image_radius

        bg_frame = numpy.asarray(frame0)
        self.height, self.width = bg_frame.shape
        assert bg_frame.dtype == numpy.uint8
        self.timestamp0 = timestamp0

        self.file.write(struct.pack(FMT[1].HEADER,
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
        self.file.write(struct.pack(FMT[1].CHUNKHEADER, timestamp, n_pts ))
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
            ymax = min( ymax, self.height)
            if ymax == self.height:
                ymin = self.height - (2*self.image_radius)

            assert ymax-ymin == (2*self.image_radius)
            assert xmax-xmin == (2*self.image_radius)

            roi = origframe[ ymin:ymax, xmin:xmax ]
            this_str_buf = roi.tostring()
            this_str_head = struct.pack(FMT[1].SUBHEADER, xmin, ymin)

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

class UfmfSaverV2(UfmfSaverBase):
    """class to write (save) .ufmf v2 files"""
    def __init__(self, file,
                 coding='MONO8',
                 frame0=None,
                 timestamp0=None,
                 max_width=np.inf,
                 max_height=np.inf,
                 xinc_yinc=None,
                 ):
        super(UfmfSaverV2,self).__init__(2)
        if hasattr(file,'write'):
            # file-like object
            self.file = file
            self._file_opened = False
        else:
            self.file = open(file,mode="w+b")
            self._file_opened = True
        buf = struct.pack( FMT[2].HEADER,
                           self.version, len(coding) )
        self.file.write(buf)
        self.file.write(coding)
        self.max_width=max_width
        self.max_height=max_height
        if frame0 is not None or timestamp0 is not None:
            self.add_keyframe('frame0',frame0,timestamp0)
        if xinc_yinc is None:
            if coding=='MONO8':
                xinc_yinc = (1,1)
            elif coding.startswith('MONO8:'):
                # Bayer pattern
                xinc_yinc = (2,2)
            elif coding=='YUV422':
                xinc_yinc = (4,1)
            else:
                warnings.warn('ufmf xinc_yinc set (1,1) because coding unknown')
        self.xinc, self.yinc = xinc_yinc

    def add_keyframe(self,keyframe_type,image_data,timestamp):
        char2 = len(keyframe_type)
        np_image_data = numpy.asarray(image_data)
        if np_image_data.dtype == np.uint8:
            dtype = 'B'
        elif np_image_data.dtype == np.float32:
            dtype = 'f'
        else:
            raise ValueError('dtype %s not supported'%image_data.dtype)
        assert np_image_data.ndim == 2
        height, width = np_image_data.shape
        assert np_image_data.strides[0] == width
        assert np_image_data.strides[1] == 1
        b =  chr(KEYFRAME_CHUNK) + chr(char2) + keyframe_type # chunkid, len(type), type
        b += struct.pack(FMT[2].KEYFRAME2,dtype,width,height,timestamp)
        self.file.write(b)
        self.file.write(buffer(np_image_data))

    def add_frame(self,origframe,timestamp,point_data):
        n_pts = len(point_data)
        b = chr(FRAME_CHUNK) + struct.pack(FMT[2].POINTS1, timestamp, n_pts)
        self.file.write(b)
        str_buf = []
        origframe = np.asarray(origframe)
        origframe_h, origframe_w = origframe.shape
        if len(point_data):
            for this_point_data in point_data:
                xidx, yidx, w, h = this_point_data[:4]
                w_radius = w//2
                h_radius = h//2

                xmin = int(round(xidx-w_radius)//self.xinc*self.xinc) # keep 2x2 Bayer
                xmin = max(0,xmin)

                xmax = xmin + w
                newxmax = min( xmax, self.max_width, origframe_w)
                if newxmax != xmax:
                    #xmin = newxmax - w
                    w = newxmax - xmin
                    xmax = newxmax

                ymin = int(round(yidx-h_radius)//self.yinc*self.yinc) # keep 2x2 Bayer
                ymin = max(0,ymin)

                ymax = ymin + h
                newymax = min( ymax, self.max_height, origframe_h)
                if newymax != ymax:
                    #ymin = newymax - h
                    h = newymax - ymin
                    ymax = newymax

                assert ymax-ymin == h
                assert xmax-xmin == w

                roi = origframe[ ymin:ymax, xmin:xmax ]
                this_str_buf = roi.tostring()
                assert len(this_str_buf)==w*h
                this_str_head = struct.pack(FMT[2].POINTS2, xmin, ymin, w, h)

                str_buf.append( this_str_head + this_str_buf )
            fullstr = ''.join(str_buf)
            self.file.write(fullstr)
        self.last_timestamp = timestamp

    def close(self):
        if self._file_opened:
            self.file.close()
            self._file_opened = False
