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

class UfmfError(Exception):
    pass

class ShortUFMFFileError(UfmfError):
    pass

class CorruptIndexError(UfmfError):
    pass

class BaseDict(dict):
    def __getattr__(self,name):
        return self[name]
    def __setattr__(self,name,value):
        self[name]=value

FMT = {1:BaseDict(HEADER = '<IIdII', # version, ....
                  CHUNKHEADER = '<dI',
                  SUBHEADER = '<II',
                  TIMESTAMP = 'd', # XXX struct.pack('<d',nan) dies
                  ),

       3:BaseDict(HEADER = '<4sIQHHB', # 'ufmf', version, index location,
                                       # max w, max h, raw coding string length
                  CHUNKID = '<B', # 0 = keyframe, 1 = points
                  KEYFRAME1 = '<B', # (type name)
                  KEYFRAME2 = '<cHHd', # (dtype, width,height,timestamp)
                  POINTS1 = '<dH', # timestamp, n_pts
                  POINTS2 = '<HHHH', # x0, y0, w, h
                  ),

       # Format 2 is the same as format 3, but the index must be at < 4 GB
       2:BaseDict(HEADER = '<4sILHHB', # 'ufmf', version, index location,
                                       # max w, max h, raw coding string length
                  CHUNKID = '<B', # 0 = keyframe, 1 = points
                  KEYFRAME1 = '<B', # (type name)
                  KEYFRAME2 = '<cHHd', # (dtype, width,height,timestamp)
                  POINTS1 = '<dH', # timestamp, n_pts
                  POINTS2 = '<HHHH', # x0, y0, w, h
                  ),
       }

KEYFRAME_CHUNK = 0
FRAME_CHUNK = 1
INDEX_DICT_CHUNK = 2

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
    VERSION_FMT = '<I' # always the first bytes
    version_buflen = struct.calcsize(VERSION_FMT)
    version_buf = fd.read( version_buflen )
    had_marker=False
    if version_buf=='ufmf':
        version_buf = fd.read( version_buflen )
        had_marker=True
    version, = struct.unpack(VERSION_FMT, version_buf)
    if version>1:
        if not had_marker:
            raise ValueError('ill-formed .ufmf file')
    if version==1:
        if had_marker:
            raise ValueError('ill-formed .ufmf file')
    fd.close()
    return version

def _write_dict(fd,save_dict):
    fd.write('d')
    fd.write(chr(len(save_dict.keys())))
    keys = save_dict.keys()
    keys.sort() # keep ordering fixed to file remains same if re-indexed
    for key in keys:
        value = save_dict[key]
        b = struct.pack('<H',len(key))
        b += key
        fd.write(b)
        if isinstance(value,dict):
            _write_dict(fd,value)
            continue
        if isinstance(value,list) or isinstance(value,np.ndarray):
            larr = np.array(value)
            assert larr.ndim==1
            dtype_char = larr.dtype.char
            bytes_per_element = larr.dtype.itemsize
            b = 'a'+dtype_char+struct.pack('<L',len(larr)*bytes_per_element)
            b += larr.tostring()
            fd.write(b)
            continue
        raise ValueError("don't know how to save value %s"%(value,))

def _read_min_chars(fd,keylen,buf_remaining):
    if len(buf_remaining)>=keylen:
        x = buf_remaining[:keylen]
        buf_remaining = buf_remaining[keylen:]
        return x,buf_remaining
    else:
        readlen = max(4096,keylen)
        newbuf = fd.read(readlen)
        buf_remaining = buf_remaining+newbuf
        return _read_min_chars(fd,keylen,buf_remaining)

def _read_array(fd,buf_remaining):
    x,buf_remaining = _read_min_chars(fd,1,buf_remaining)
    dtype_char = x

    n_bytes_calcsize = struct.calcsize('<L')
    x,buf_remaining = _read_min_chars(fd,n_bytes_calcsize,buf_remaining)
    n_bytes_buf=x
    n_bytes, = struct.unpack('<L',n_bytes_buf)

    data_buf,buf_remaining = _read_min_chars(fd,n_bytes,buf_remaining)
    larr = np.fromstring(data_buf,dtype=dtype_char)
    return larr, buf_remaining

def _read_dict(fd,buf_remaining=None):
    if buf_remaining is None:
        buf_remaining=''
    x,buf_remaining = _read_min_chars(fd,1,buf_remaining)
    n_keys = ord(x)
    result = {}
    Hsize = struct.calcsize('<H')
    for key_num in range(n_keys):
        x,buf_remaining = _read_min_chars(fd,Hsize,buf_remaining)
        keylen, = struct.unpack('<H',x)
        x,buf_remaining = _read_min_chars(fd,keylen,buf_remaining)
        key = x
        x,buf_remaining = _read_min_chars(fd,1,buf_remaining)
        id = x
        if id=='d':
            value,buf_remaining = _read_dict(fd,buf_remaining)
        elif id=='a':
            value,buf_remaining = _read_array(fd,buf_remaining)
        else:
            raise ValueError("don't know how to read value with id %s"%(id,))
        assert key not in value
        result[key] = value
    return result,buf_remaining

def minimize_rectangle_coverage( overlap_rects, rectangle_penalty=0 ):
    """reduce overlap in set of rectangles covering the union of input rects

    rectangle_penalty - a penalty for each additional rectangle, used
      to discourage small rectangles

    See, for example:

    Heinrich-Litan and Luebbecke (2006), Rectangle
    covers revisited computationally, J. Exp. Algorithmics

    Ferrari, Sankar, Sklansky (1984) Minimal Rectangular Partitions of
    Digitized Blobs. Computer Vision, Graphics, and Image Processing.

    """
    warnings.warn('not implemented: minimize_rectangle_coverage')
    return overlap_rects

def Ufmf(filename,**kwargs):
    """factory function to return UfmfBase class instance"""
    version = identify_ufmf_version(filename)
    if version==1:
        return UfmfV1(filename,**kwargs)
    elif version==2:
        if 'seek_ok' in kwargs:
            kwargs.pop('seek_ok')
        return UfmfV2(filename,**kwargs)
    elif version==3:
        if 'seek_ok' in kwargs:
            kwargs.pop('seek_ok')
        return UfmfV3(filename,**kwargs)
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

class _UFmfV3LowLevelReader(object):
    def __init__(self,fd,version):
        self._fd = fd
        self._version = version

        self._keyframe2_sz = struct.calcsize(FMT[self._version].KEYFRAME2)
        self._points1_sz =   struct.calcsize(FMT[self._version].POINTS1)
        self._points2_sz =   struct.calcsize(FMT[self._version].POINTS2)

    def _read_keyframe_chunk(self):
        """read keyframe chunk from just after chunk_id byte

        start_location is the file locate at which the chunk started.
        (i.e. curpos-1)
        """
        len_type = ord(self._fd_read(1))
        keyframe_type = self._fd_read(len_type)
        assert len(keyframe_type)==len_type
        intup = struct.unpack(FMT[self._version].KEYFRAME2,
                              self._fd_read(self._keyframe2_sz))
        dtype_char,width,height,timestamp=intup

        if dtype_char=='B':
            dtype=np.uint8
            sz=1
        elif dtype_char=='f':
            dtype=np.float32
            sz=4
        else:
            raise ValueError('unknown dtype char "%s" (0x%0x)'%(dtype_char,
                                                                ord(dtype_char)))
        read_len = width*height*sz
        buf = self._fd_read(read_len)
        frame = np.fromstring(buf,dtype=dtype)
        frame.shape = (height,width)
        return keyframe_type,frame,timestamp

    def _read_frame_chunk(self):
        """read frame chunk from just after chunk_id byte

        start_location is the file locate at which the chunk started
        (i.e. curpos-1)
        """
        intup = struct.unpack(FMT[self._version].POINTS1,
                              self._fd_read(self._points1_sz))
        timestamp, n_pts = intup

        regions = []
        for ptno in range(n_pts):
            intup = struct.unpack(FMT[self._version].POINTS2,
                                  self._fd_read(self._points2_sz))
            (xmin, ymin, w, h) = intup
            lenbuf = w*h
            buf = self._fd_read(lenbuf)
            im = np.fromstring(buf,dtype=np.uint8)
            im.shape = (h,w)
            regions.append( (xmin,ymin,im) )
        return timestamp,regions

    def _fd_read(self,n_bytes,short_OK=False):
        buf = self._fd.read(n_bytes)
        if len(buf)!=n_bytes:
            if not short_OK:
                raise ShortUFMFFileError('expected %d bytes, got %d: short file %s?'%(
                    n_bytes,len(buf), self._fd.name))
        return buf

class _UFmfV3Indexer(object):
    """create an index from an un-unindexed .ufmf v3 file"""
    def __init__(self,fd, version,
                 ignore_preexisting_index=False,
                 short_file_ok=False,
                 index_progress=False,
                 ):
        self._fd = fd
        self._version = version

        self._keyframe2_sz = struct.calcsize(FMT[self._version].KEYFRAME2)
        self._points1_sz = struct.calcsize(FMT[self._version].POINTS1)
        self._points2_sz = struct.calcsize(FMT[self._version].POINTS2)

        self.r = _UFmfV3LowLevelReader(self._fd, self._version)
        self._short_file_ok = short_file_ok
        self._index_chunk_location = None
        self._ignore_preexisting_index = ignore_preexisting_index
        self._index_progress = index_progress
        self._create_index()

    def get_index(self):
        result = {'frame':self._index['frame']}
        # remove defaultdict and convert to dict
        result['keyframe'] = {}
        for keyframe_type,value in self._index['keyframe'].iteritems():
            result['keyframe'][keyframe_type] = value
        return result

    def get_expected_index_chunk_location(self):
        return self._index_chunk_location

    def _create_index(self):
        self._index = {'keyframe':collections.defaultdict(dict),
                       'frame':dict(timestamp=[],
                                    loc=[])}
        if self._index_progress:
            import progressbar

            widgets=['indexing: ', progressbar.Percentage(), ' ',
                     progressbar.Bar(), ' ', progressbar.ETA()]
            cur_pos = self._fd.tell()
            # find file size (use seek not stat because we may not know name)
            try:
                self._fd.seek(0,os.SEEK_END) # last byte
                final_pos = self._fd.tell()
            finally:
                self._fd.seek(cur_pos)
            pbar=progressbar.ProgressBar(widgets=widgets,maxval=final_pos).start()
        while 1:
            if self._index_progress:
                pbar.update(self._fd.tell())
            chunk_id, result = self._index_next_chunk()
            if chunk_id is None:
                break # no more frames
        if self._index_progress:
            pbar.finish()
        # convert to arrays
        for keyframe_type in self._index['keyframe'].keys():
            for key in self._index['keyframe'][keyframe_type].keys():
                self._index['keyframe'][keyframe_type][key]=np.array(
                    self._index['keyframe'][keyframe_type][key])
        for key in self._index['frame'].keys():
            self._index['frame'][key]=np.array(
                self._index['frame'][key])
        if self._index_chunk_location is None:
            self._index_chunk_location = self._fd.tell()

    def _index_next_chunk(self):
        loc = self._fd.tell()
        chunk_id_str = self.r._fd_read(1,short_OK=True)
        if chunk_id_str == '':
            return None, None
        chunk_id = ord(chunk_id_str)
        try:
            if chunk_id==KEYFRAME_CHUNK:
                result = self.r._read_keyframe_chunk()
                keyframe_type,frame,timestamp=result
                tmp = self._index['keyframe'][keyframe_type]
                if len(tmp)==0:
                    tmp['timestamp']=[]
                    tmp['loc']=[]
                tmp['timestamp'].append(timestamp)
                tmp['loc'].append(loc)
            elif chunk_id==FRAME_CHUNK:
                # read frame chunk
                result = self.r._read_frame_chunk()
                timestamp,regions = result
                tmp = self._index['frame']
                tmp['timestamp'].append(timestamp)
                tmp['loc'].append(loc)
            elif chunk_id==INDEX_DICT_CHUNK:
                if not self._ignore_preexisting_index:
                    raise PreexistingIndexExists('indexer encountered index',
                                                 loc=loc)
                else:
                    self.r._fd.seek(-1,os.SEEK_CUR) # undo chunk read - go back
                    return None,None
            else:
                raise ValueError('unexpected byte %d where chunk ID expected'%(
                    chunk_id,))
        except ShortUFMFFileError:
            if self._short_file_ok:
                self._index_chunk_location = loc
                return None,None
            else:
                raise
        return chunk_id, result

class PreexistingIndexExists(Exception):
    def __init__(self,mystr,loc=None):
        super(PreexistingIndexExists,self).__init__(mystr)
        self.loc = loc

class UfmfV3(UfmfBase):
    """class to read .ufmf version 3 files"""
    def _get_interface_version(self):
        return 3

    def __init__(self,file,
                 mode='rb',
                 ignore_preexisting_index=False,
                 is_ok_to_write_regenerated_index=True,
                 short_file_ok=False,
                 raise_write_errors=False,
                 index_progress=False,
                 ):
        """
        **Arguments**
        file : file name or file object
            The file with the .ufmf data to read

        **Optional keyword arguments**
        mode : string
            The mode to open the file with (e.g. 'rb' for read only, binary)
        ignore_preexisting_index : boolean
            Whether to ignore the index generate a new one
        is_ok_to_write_regenerated_index : boolean
            Whether to write index to disk if regenerated
        short_file_ok : boolean
            Whether to ignore short file errors
        raise_write_errors : boolean
            Whether to ignore short file errors
        index_progress : boolean
            Whether to display a text-based progressbar while indexing
        """
        super(UfmfV3,self).__init__()
        if hasattr(file,'read'):
            # file-like object
            self._file_opened=False
            self._fd = file
        else:
            # filename
            stat_result = os.stat(file)
            self._fd = open(file,mode=mode)
            self._file_opened=True

        expected_version = self._get_interface_version()
        self._r = _UFmfV3LowLevelReader(self._fd,expected_version)

        bufsz = struct.calcsize(FMT[expected_version].HEADER)
        buf = self._r._fd_read( bufsz )
        intup = struct.unpack(FMT[expected_version].HEADER, buf)
        (ufmf_str, self._version, index_location,
         self._max_width, self._max_height,
         coding_str_len) = intup
        assert ufmf_str=='ufmf'
        assert expected_version==self._version
        self._coding = self._r._fd_read( coding_str_len )
        self._next_frame = 0
        if ignore_preexisting_index:
            index_location = 0

        self._keyframe2_sz = struct.calcsize(FMT[self._version].KEYFRAME2)
        self._points1_sz = struct.calcsize(FMT[self._version].POINTS1)
        self._points2_sz = struct.calcsize(FMT[self._version].POINTS2)

        if index_location == 0:
            # no pre-existing index. generate it.
            # save it.
            tmp = _UFmfV3Indexer(
                self._fd, self._version,
                ignore_preexisting_index=ignore_preexisting_index,
                short_file_ok=short_file_ok,
                index_progress=index_progress,
                )
            self._index = tmp.get_index()
            if not is_ok_to_write_regenerated_index:
                return
            loc = tmp.get_expected_index_chunk_location()
            self._fd.seek(loc)
            try:
                index_dict_location = self._fd.tell()+1 # not the chunk location - add one
                if self._version==2 and index_dict_location > 4294967295:
                    raise ValueError('index location will not fit in .ufmf v2 file')

                b = chr(INDEX_DICT_CHUNK)
                self._fd.write(b)
                _write_dict( self._fd, self._index )
                self._fd.truncate()
                self._fd.seek(0)
                buf = struct.pack( FMT[self._version].HEADER, 'ufmf',
                                   self._version, index_dict_location,
                                   self._max_width, self._max_height,
                                   len(self._coding) )
                self._fd.write(buf)
            except IOError, err:
                if raise_write_errors:
                    raise
                else:
                    warnings.warn('IO error when trying to save .ufmf index '
                                  'for %s (was .ufmf file opened read-only?)'%(
                        file,))
        else:
            # we are already in the chunk, so we don't expect INDEX_DICT_CHUNK
            # read index
            self._seek(index_location)
            buf = self._fd.read(4096)

            # next char is 'd' for dict
            id = buf[:1]
            buf = buf[1:]
            try:
                assert id=='d' # dictionary

                self._index,buf_remaining = _read_dict(self._fd, buf_remaining=buf)
                if len(buf_remaining)!=0:
                    raise ValueError('bytes after expected end of file')
            except:
                raise CorruptIndexError('the .ufmf index is corrupt. '
                                        '(Hint: Try the ufmf_reindex command.)')

    def get_index(self):
        return self._index

    def get_keyframe_for_timestamp(self, keyframe_type, timestamp):
        """return the most recent keyframe before or at the time of timestamp"""
        ts = self._index['keyframe'][keyframe_type]['timestamp']
        cond = timestamp >= ts
        idxs = np.nonzero(cond)[0]
        if len(idxs)==0:
            raise NoMoreFramesException('no keyframe_type %s prior to %s'%(
                keyframe_type,repr(timestamp)))
        idx = idxs[-1]
        return self._get_keyframe_N(keyframe_type,idx)

    def get_number_of_frames(self):
        """return the number of frames"""
        return len(self._index['frame']['loc'])

    def get_max_size(self):
        return (self._max_width, self._max_height)

    def get_coding(self):
        return self._coding

    def get_progress(self):
        locs = self._index['frame']['loc']
        return float(self._next_frame)/len(locs)

    def set_next_frame(self,fno):
        self._next_frame = fno

    def readframes(self):
        locs = self._index['frame']['loc']
        while self._next_frame < len(locs):
            loc = locs[self._next_frame]
            self._next_frame += 1

            self._seek(loc+1)
            result = self._r._read_frame_chunk()
            timestamp,regions = result
            yield timestamp,regions

    def _get_keyframe_N(self, keyframe_type, N):
        """get Nth keyframe of type keyframe_type"""
        try:
            tmp = self._index['keyframe'][keyframe_type]
        except KeyError:
            raise NoMoreFramesException('no keyframe_type %s'%keyframe_type)
        loc = tmp['loc'][N]
        timestamp = tmp['timestamp'][N]

        self._seek(loc+1)
        result = self._r._read_keyframe_chunk()
        test_keyframe_type,frame,test_timestamp=result
        assert keyframe_type==test_keyframe_type
        assert timestamp==test_timestamp
        return frame,timestamp

    def _seek(self,loc):
        self._fd.seek(loc,0)

    def get_bg_image(self):
        """return the first raw image (for compatability with UfmfV1)"""
        return self._get_keyframe_N('frame0',0)

    def close(self):
        if self._file_opened:
            self._fd.close()
            self._file_opened = False

class UfmfV2(UfmfV3):
    """class to read .ufmf version 2 files"""
    def _get_interface_version(self):
        return 2

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
        self._last_frame = None
        self.filename = filename
        try:
            self._bg0,self._ts0=self._ufmf.get_bg_image()
        except NoMoreFramesException:
            pass
        self._darken=darken
        self._allow_no_such_frame_errors = allow_no_such_frame_errors
        self._timestamps = None
        if isinstance(self._ufmf,UfmfV1):
            self.format = 'MONO8' # by definition
            self._fno2loc = None
            if self._ufmf.use_conventional_named_mean_fmf:
                assert white_background==False
        self.white_background = white_background
        self.abs_diff = abs_diff
        if self.abs_diff:
            if not (isinstance(self._ufmf,UfmfV1) and
                    self._ufmf.use_conventional_named_mean_fmf):
                raise NotImplementedError('abs_diff currently requires UfmfV1 '
                                          'and use_conventional_named_mean_fmf')

    def close(self):
        self._ufmf.close()

    def get_n_frames(self):
        if isinstance(self._ufmf,UfmfV1):
            self._fill_timestamps_and_locs()
            last_frame = len(self._fno2loc)
            return last_frame+1
        else:
            return self._ufmf.get_number_of_frames()

    def get_format(self):
        if isinstance(self._ufmf,UfmfV1):
            return self.format
        else:
            return self._ufmf.get_coding()

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
        if isinstance(self._ufmf,UfmfV1):
            if 0<= fno < len(self._fno2loc):
                loc = self._fno2loc[fno]
                self._ufmf.seek(loc)
                self._last_frame = None
            else:
                raise NoSuchFrameError('fno %d not in .ufmf file'%fno)
        else:
            self._ufmf.set_next_frame( fno )

    def get_mean_for_timestamp(self, timestamp):
        if isinstance(self._ufmf,UfmfV1):
            return self._ufmf.get_mean_for_timestamp( timestamp )
        else:
            mean_im, tmp_timestamp = self._ufmf.get_keyframe_for_timestamp(
                'mean', timestamp )
            return mean_im

    def get_next_frame(self, _return_more=False):
        have_frame = False
        more = {}
        for timestamp, regions in self._ufmf.readframes():
            if isinstance(self._ufmf,UfmfV1):
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
            else:
                try:
                    mean_image,im_timestamp=self._ufmf.get_keyframe_for_timestamp('mean',timestamp)
                except (KeyError, NoMoreFramesException):
                    warnings.warn('UfmfV3 fmf emulator filling bg with white')
                    w,h=self._ufmf.get_max_size()
                    mean_image = numpy.empty((h,w),dtype=np.uint8)
                    mean_image.fill(255)
                if _return_more:
                    sumsqf_image,sq_timestamp=self._ufmf.get_keyframe_for_timestamp('sumsq',timestamp)
                    more['sumsqf'] = sumsqf_image
                if not self.white_background:
                    self._last_frame = np.array(mean_image,copy=True).astype(np.uint8)
                else:
                    self._last_frame = np.empty(mean_image.shape,dtype=np.uint8)
                    self._last_frame.fill(255)
                more['mean'] = mean_image
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

        if isinstance(self._ufmf,UfmfV3):
            index = self._ufmf.get_index()
            self._timestamps = index['frame']['timestamp']
            return

        assert isinstance(self._ufmf,UfmfV1)

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
        if isinstance(self._ufmf,UfmfV1):
            return self._bg0.shape[0]
        else:
            return self._ufmf.get_max_size()[1]

    def get_width(self):
        if isinstance(self._ufmf,UfmfV1):
            return self._bg0.shape[1]
        else:
            return self._ufmf.get_max_size()[0]

def UfmfSaver( file,
               frame0=None,
               timestamp0=None,
               **kwargs):
    """factory function to return UfmfSaverBase instance"""
    default_version = 3
    version = kwargs.pop('version',default_version)
    if version is None:
        version = default_version

    if version==1:
        return UfmfSaverV1(file,frame0,timestamp0,**kwargs)
    elif version==2:
        return UfmfSaverV2(file,frame0=frame0,timestamp0=timestamp0,**kwargs)
    elif version==3:
        return UfmfSaverV3(file,frame0=frame0,timestamp0=timestamp0,**kwargs)
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

class UfmfSaverV3(UfmfSaverBase):
    """class to write (save) .ufmf v3 files"""
    def _get_interface_version(self):
        return 3

    def __init__(self, file,
                 coding='MONO8',
                 frame0=None,
                 timestamp0=None,
                 max_width=None,
                 max_height=None,
                 xinc_yinc=None,
                 ):
        super(UfmfSaverV3,self).__init__(self._get_interface_version())
        if hasattr(file,'write'):
            # file-like object
            self.file = file
            self._file_opened = False
        else:
            self.file = open(file,mode="w+b")
            self._file_opened = True
        self.coding = coding
        if max_width is None or max_height is None:
            raise ValueError('max_width and max_height must be set')
        self.max_width=max_width
        self.max_height=max_height
        buf = struct.pack( FMT[self.version].HEADER, 'ufmf',
                           self.version, 0, self.max_width, self.max_height,
                           len(self.coding) )
        self.file.write(buf)
        self.file.write(coding)
        if xinc_yinc is None:
            if coding=='MONO8':
                xinc_yinc = (2,2)
            elif coding.startswith('MONO8:'):
                # Bayer pattern
                xinc_yinc = (2,2)
            elif coding=='YUV422':
                xinc_yinc = (4,1)
            else:
                warnings.warn('ufmf xinc_yinc set (2,2) because coding unknown')
                xinc_yinc = (2,2)
        self.xinc, self.yinc = xinc_yinc
        self._index = {'keyframe':collections.defaultdict(dict),
                       'frame':{}}
        if frame0 is not None or timestamp0 is not None:
            self.add_keyframe('frame0',frame0,timestamp0)

        self.min_bytes = struct.calcsize(FMT[self.version].POINTS2)

    def add_keyframe(self,keyframe_type,image_data,timestamp):
        char2 = len(keyframe_type)
        np_image_data = numpy.asarray(image_data)
        if np_image_data.dtype == np.uint8:
            dtype = 'B'
            strides1=1
        elif np_image_data.dtype == np.float32:
            dtype = 'f'
            strides1=4
        else:
            raise ValueError('dtype %s not supported'%image_data.dtype)
        assert np_image_data.ndim == 2
        height, width = np_image_data.shape
        try:
            assert np_image_data.strides[0] == width*np_image_data.strides[1]
            assert np_image_data.strides[1] == strides1
        except:
            print 'np_image_data.strides, width',np_image_data.strides, width
            raise
        b =  chr(KEYFRAME_CHUNK) + chr(char2) + keyframe_type # chunkid, len(type), type
        b += struct.pack(FMT[self.version].KEYFRAME2,dtype,width,height,timestamp)
        loc = self.file.tell()
        tmp = self._index['keyframe'][keyframe_type]
        if len(tmp)==0:
            tmp['timestamp']=[]
            tmp['loc']=[]
        tmp['timestamp'].append(timestamp)
        tmp['loc'].append(loc)
        self.file.write(b)
        self.file.write(buffer(np_image_data))

    def _add_frame_regions(self,timestamp,regions):
        n_pts = len(regions)
        b = chr(FRAME_CHUNK) + struct.pack(FMT[self.version].POINTS1, timestamp, n_pts)
        loc = self.file.tell()
        tmp = self._index['frame']
        if len(tmp)==0:
            tmp['timestamp']=[]
            tmp['loc']=[]
        tmp['timestamp'].append(timestamp)
        tmp['loc'].append(loc)
        self.file.write(b)
        str_buf = []
        for region in regions:
            xmin,ymin,roi = region
            h,w = roi.shape
            this_str_buf = roi.tostring()
            assert len(this_str_buf)==w*h
            this_str_head = struct.pack(FMT[self.version].POINTS2, xmin, ymin, w, h)
            str_buf.append( this_str_head + this_str_buf )
        fullstr = ''.join(str_buf)
        self.file.write(fullstr)

    def add_frame(self,origframe,timestamp,point_data):
        origframe = np.asarray(origframe)
        origframe_h, origframe_w = origframe.shape
        rects = []
        regions = []
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

                rects.append( (xmin,ymin, xmax,ymax) )

            ## rects = minimize_rectangle_coverage(
            ##     rects, rectangle_penalty=self.min_bytes )

            for (xmin,ymin,xmax,ymax) in rects:
                roi = origframe[ ymin:ymax, xmin:xmax ]
                regions.append( (xmin, ymin, roi) )

        self._add_frame_regions(timestamp,regions)
        self.last_timestamp = timestamp
        return rects

    def close(self):
        b = chr(INDEX_DICT_CHUNK)
        self.file.write(b)
        loc = self.file.tell()
        if self.version==2 and loc > 4294967295:
            raise ValueError('index location will not fit in .ufmf v2 file')
        _write_dict(self.file,self._index)
        self.file.seek(0)
        buf = struct.pack( FMT[self.version].HEADER, 'ufmf',
                           self.version, loc,
                           self.max_width, self.max_height,
                           len(self.coding) )
        self.file.write(buf)
        if self._file_opened:
            self.file.close()
            self._file_opened = False

class AutoShrinkUfmfSaverV3(UfmfSaverV3):
    """class to write (save) .ufmf v3 files without useless keyframes

    Assumes that frames and keyframes will come in temporal order.
    Does not save keyframes covering intervals in which no frames arrived.
    """
    def __init__(self,*args,**kwargs):
        self._cached_keyframes = {}
        super(AutoShrinkUfmfSaverV3,self).__init__(*args,**kwargs)
    def _add_frame_regions(self,timestamp,regions):
        if len(regions):
            for kf_type in self._cached_keyframes.keys():
                kf_image_data, kf_timestamp = self._cached_keyframes[kf_type]
                super(AutoShrinkUfmfSaverV3,self).add_keyframe( \
                                          kf_type, kf_image_data, kf_timestamp)
                del self._cached_keyframes[kf_type]
        super(AutoShrinkUfmfSaverV3,self)._add_frame_regions(timestamp,regions)
    def add_keyframe(self,keyframe_type,image_data,timestamp):
        self._cached_keyframes[keyframe_type] = (image_data,timestamp)

class UfmfSaverV2(UfmfSaverV3):
    """class to write (save) .ufmf v2 files"""
    def _get_interface_version(self):
        return 2
