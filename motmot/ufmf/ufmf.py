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
CHUNKHEADER_FMT = '<II'
SUBHEADER_FMT = '<II'
TIMESTAMP_FMT = 'd' # XXX struct.pack('<d',nan) dies


class NoMoreFramesException( Exception ):
    pass

class InvalidMovieFileException( Exception ):
    pass

## class FlyMovie:

##     def __init__(self, filename,check_integrity=False):
##         self.filename = filename
##         try:
##             self.file = open(self.filename,mode="r+b")
##         except IOError:
##             self.file = open(self.filename,mode="r")
##             self.writeable = False
##         else:
##             self.writeable = True

##         r=self.file.read # shorthand
##         t=self.file.tell # shorthand
##         size=struct.calcsize
##         unpack=struct.unpack

##         version_buf = r(size(VERSION_FMT))
##         if len(version_buf)!=size(VERSION_FMT):
##             raise InvalidMovieFileException("could not read data file")

##         version, = unpack(VERSION_FMT,version_buf)
##         if version not in (1,3):
##             raise NotImplementedError('Can only read version 1 and 3 files')

##         if version  == 1:
##             self.format = 'MONO8'
##             self.bits_per_pixel = 8
##         elif version == 3:
##             format_len = unpack(FORMAT_LEN_FMT,r(size(FORMAT_LEN_FMT)))[0]
##             self.format = r(format_len)
##             self.bits_per_pixel = unpack(BITS_PER_PIXEL_FMT,r(size(BITS_PER_PIXEL_FMT)))[0]

##         try:
##             self.framesize = unpack(FRAMESIZE_FMT,r(size(FRAMESIZE_FMT)))
##         except struct.error:
##             raise InvalidMovieFileException('file could not be read')

##         self.bytes_per_chunk, = unpack(CHUNKSIZE_FMT,r(size(CHUNKSIZE_FMT)))
##         self.n_frames, = unpack(N_FRAME_FMT,r(size(N_FRAME_FMT)))
##         self.timestamp_len = size(TIMESTAMP_FMT)
##         self.chunk_start = self.file.tell()
##         self.next_frame = None

## 	if self.n_frames == 0: # unknown movie length, read to find out
##             # seek to end of the movie
##             self.file.seek(0,2)
##             # get the byte position
##             eb = self.file.tell()
##             # compute number of frames using bytes_per_chunk
##             self.n_frames = int(math.ceil((eb-self.chunk_start)/self.bytes_per_chunk))
##             # seek back to the start
##             self.file.seek(self.chunk_start,0)

##         if check_integrity:
##             n_frames_ok = False
##             while not n_frames_ok:
##                 try:
##                     self.get_frame(-1)
##                     n_frames_ok = True
##                 except NoMoreFramesException:
##                     self.n_frames -= 1
##                     if self.n_frames == 0:
##                         break
## 	    self.file.seek(self.chunk_start) # go back to beginning

##         self._all_timestamps = None # cache

##     def close(self):
##         self.file.close()
##         self.writeable = False
##         self.n_frames = None
##         self.next_frame = None

##     def get_width(self):
##         """returns width of data

##         to get width of underlying image:

##           image_width = fmf.get_width()//(fmf.get_bits_per_pixel()//8)
##         """
##         return self.framesize[1]

##     def get_height(self):
##         return self.framesize[0]

##     def get_n_frames(self):
##         return self.n_frames

##     def get_format(self):
##         return self.format

##     def get_bits_per_pixel(self):
##         return self.bits_per_pixel

##     def _read_next_frame(self,allow_partial_frames=False):
##         data = self.file.read( self.bytes_per_chunk )
##         if data == '':
##             raise NoMoreFramesException('EOF')
##         if len(data)<self.bytes_per_chunk:
##             if allow_partial_frames:
##                 missing_bytes = self.bytes_per_chunk-len(data)
##                 data = data + '\x00'*missing_bytes
##                 warnings.warn('appended %d bytes (to a total of %d), image will be corrupt'%(missing_bytes,self.bytes_per_chunk))
##             else:
##                 raise NoMoreFramesException('short frame '\
##                                             '(%d bytes instead of %d)'%(
##                     len(data),self.bytes_per_chunk))
##         timestamp_buf = data[:self.timestamp_len]
##         timestamp, = struct.unpack(TIMESTAMP_FMT,timestamp_buf)

##         frame = numpy.fromstring(data[self.timestamp_len:],numpy.uint8)
##         frame.shape = self.framesize

## ##        if self.format == 'MONO8':
## ##            frame = numpy.fromstring(data[self.timestamp_len:],numpy.uint8)
## ##            frame.shape = self.framesize
## ##        elif self.format in ('YUV411','YUV422'):
## ##            frame = numpy.fromstring(data[self.timestamp_len:],numpy.uint16)
## ##            frame.shape = self.framesize
## ##        elif self.format in ('MONO16',):
## ##            print 'self.framesize',self.framesize
## ##            frame = numpy.fromstring(data[self.timestamp_len:],numpy.uint8)
## ##            frame.shape = self.framesize
## ##        else:
## ##            raise NotImplementedError("Reading not implemented for %s format"%(self.format,))
##         return frame, timestamp

##     def _read_next_timestamp(self):
##         read_len = struct.calcsize(TIMESTAMP_FMT)
##         timestamp_buf = self.file.read( read_len )
##         self.file.seek( self.bytes_per_chunk-read_len, 1) # seek to next frame
##         if timestamp_buf == '':
##             raise NoMoreFramesException('EOF')
##         timestamp, = struct.unpack(TIMESTAMP_FMT,timestamp_buf)
##         return timestamp

##     def is_another_frame_available(self):
##         try:
##             if self.next_frame is None:
##                 self.next_frame = self._read_next_frame()
##         except NoMoreFramesException:
##             return False
##         return True

##     def get_next_frame(self,allow_partial_frames=False):
##         if self.next_frame is not None:
##             frame, timestamp = self.next_frame
##             self.next_frame = None
##             return frame, timestamp
##         else:
##             frame, timestamp = self._read_next_frame(
##                 allow_partial_frames=allow_partial_frames)
##             return frame, timestamp

##     def get_frame(self,frame_number,allow_partial_frames=False):
##         if frame_number < 0:
##             frame_number = self.n_frames + frame_number
##         if frame_number < 0:
##             raise IndexError("negative index out of range (movie has no frames)")
##         seek_to = self.chunk_start+self.bytes_per_chunk*frame_number
##         self.file.seek(seek_to)
##         self.next_frame = None
##         return self.get_next_frame(allow_partial_frames=allow_partial_frames)

##     def get_all_timestamps(self):
##         if self._all_timestamps is None:
##             self.seek(0)
##             read_len = struct.calcsize(TIMESTAMP_FMT)
##             self._all_timestamps = []
##             while 1:
##                 timestamp_buf = self.file.read( read_len )
##                 self.file.seek( self.bytes_per_chunk-read_len, 1) # seek to next frame
##                 if timestamp_buf == '':
##                     break
##                 timestamp, = struct.unpack(TIMESTAMP_FMT,timestamp_buf)
##                 self._all_timestamps.append( timestamp )
##             self.next_frame = None
##             self._all_timestamps = numpy.asarray(self._all_timestamps)
##         return self._all_timestamps

##     def seek(self,frame_number):
##         if frame_number < 0:
##             frame_number = self.n_frames + frame_number
##         seek_to = self.chunk_start+self.bytes_per_chunk*frame_number
##         self.file.seek(seek_to)
##         self.next_frame = None

##     def get_next_timestamp(self):
##         if self.next_frame is not None:
##             frame, timestamp = self.next_frame
##             self.next_frame = None
##             return timestamp
##         else:
##             timestamp = self._read_next_timestamp()
##             return timestamp

##     def get_frame_at_or_before_timestamp(self, timestamp):
##         tss = self.get_all_timestamps()
##         at_or_before_timestamp_cond = tss <= timestamp
##         nz = numpy.nonzero(at_or_before_timestamp_cond)
##         if len(nz)==0:
##             raise ValueError("no frames at or before timestamp given")
##         fno = nz[-1]
##         return self.get_frame(fno)

class Ufmf:
    def __init__(self,filename):
        mode = "rb"
        self.file = open(filename,mode=mode)

        bufsz = struct.calcsize(HEADER_FMT)
        buf = self.file.read( bufsz )
        intup = struct.unpack(HEADER_FMT, buf)
        (self.version, self.image_radius,
         self.timestamp0,
         self.width, self.height) = intup
        
        print '(self.version, self.image_radius, self.timestamp0, self.width, self.height)',(self.version, self.image_radius, self.timestamp0, self.width, self.height)

        bg_im_buf = self.file.read( self.width*self.height)
        bg_im = numpy.fromstring( bg_im_buf, dtype=numpy.uint8)
        bg_im.shape = self.height, self.width
        self.bg_im = bg_im

    def get_full_image(self):
        return self.bg_im

    def close(self):
        self.file.close()
        
    def _dump_frames(self):
        chunkheadsz = struct.calcsize( CHUNKHEADER_FMT )
        subsz = struct.calcsize(SUBHEADER_FMT)

        chunkwidth = 2*self.image_radius
        chunkheight = 2*self.image_radius
        chunkimsize = chunkwidth*chunkheight
        while 1:
            buf = self.file.read( chunkheadsz )
            if not len(buf):
                # no more frames (EOF)
                break
            intup = struct.unpack(CHUNKHEADER_FMT, buf)
            (timestamp, n_pts) = intup
            print '*'*80
            print 'timestamp, n_pts',timestamp, n_pts
            for ptnum in range(n_pts):
                subbuf = self.file.read(subsz)
                intup = struct.unpack(SUBHEADER_FMT, subbuf)
                xmin, ymin = intup

                print 'xmin,ymin',xmin,ymin
                
                buf = self.file.read( chunkimsize )
                bufim = numpy.fromstring( buf, dtype = numpy.uint8 )
                bufim.shape = chunkheight, chunkwidth
                print 'bufim'
                print bufim
                print
            print
            print

class UfmfSaver:
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
        
            ymin = int(round(yidx-self.image_radius))
            ymin = max(0,ymin)

            ymax = ymin + 2*self.image_radius
            ymax = min( ymax, self.width)

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
        if self.timestamp0 is not None:
            # don't print twice
            dur = self.last_timestamp - self.timestamp0
            print 'saved %.1f seconds of small data'%(dur,)
            self.timestamp0 = None

    def __del__(self):
        if hasattr(self,'file'):
            self.close()
