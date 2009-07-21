import pkg_resources
import motmot.ufmf.ufmf as ufmf
import numpy
import tempfile, os

ufmf_versions = [None,1,2,3] # None = default

def test_a():
    for version in ufmf_versions:
        if version==1:
            close_its = (True,)
        else:
            close_its = (True,False)
        for close_it in close_its:
            yield check_a, version, close_it

def check_a(version,close_it):
    if 1: # indent to match old code organization
        w = 640
        h = 480

        frame1 = numpy.zeros( (h,w), dtype = numpy.uint8 )
        frame1[20:30,30:32] = 255
        frame1[20:30,30:32] = 255

        timestamp1 = 1e-20

        filename = tempfile.mkstemp()[1]
        radius = 5
        if version==1:
            kwargs = dict(image_radius=radius)
        else:
            kwargs = dict(max_width=w,
                          max_height=h,
                          )
        subw = 2*radius
        subh = 2*radius
        try:
            us = ufmf.UfmfSaver( filename,
                                 frame1,
                                 timestamp1,
                                 version=version,
                                 **kwargs)
            assert isinstance(us,ufmf.UfmfSaverBase)

            frame2 = numpy.zeros( (h,w), dtype = numpy.uint8 )


            frame2[20:30,30:40] = 1
            frame2[21,31] = 8
            frame2[60:70, 80:90] = 100
            frame2[61,81] = 7
            timestamp2 = 1

            pts = [ (35,25,subw,subh), # x,y,w,h
                    (85,65,subw,subh), # x,y,w,h
                    ]

            us.add_frame( frame2, timestamp2, pts )

            if close_it:
                us.close()

            us2 = ufmf.Ufmf(filename)
            assert isinstance(us2,ufmf.UfmfBase)
            test_frame1, test_timestamp1 = us2.get_bg_image()
            assert numpy.allclose( test_frame1, frame1 )
            assert test_timestamp1 == timestamp1
            us2.close()
        finally:
            os.unlink(filename)

def test_b():
    for seek in (True,False):
        for version in ufmf_versions:
            if version==1:
                close_its = (True,)
            else:
                close_its = (True,False)
            for close_it in close_its:
                yield check_b, seek, version, close_it

def check_b(seek_ok, version, close_it):
    if 1: # indent to match old code organization
        w = 752
        h = 240

        frame1 = numpy.zeros( (h,w), dtype = numpy.uint8 )
        frame1[20:30,30:32] = 255
        frame1[20:30,30:32] = 255

        timestamp = 1
        radius = 15

        filename = tempfile.mkstemp()[1]
        if version==1:
            kwargs = dict(image_radius=radius)
        else:
            kwargs = dict(max_width=w,
                          max_height=h,
                          )
        subw = 2*radius
        subh = 2*radius
        try:
            us = ufmf.UfmfSaver( filename,
                                 frame1,
                                 timestamp,
                                 version=version,
                                 **kwargs)
            assert isinstance(us,ufmf.UfmfSaverBase)
            frame2 = numpy.zeros( (h,w), dtype = numpy.uint8 )
            frame2[:,0::2] = range(0, w, 2) # clips (broadcast)
            for i in range(h):
                frame2[i,1::2] = i

            ll_pts = [
                [ (345, 144),
                  (521, 118),
                  ],
                [ (347, 144),
                  (522, 119),
                  (367, 229),
                  ],
                [(349,145),
                 (522,120),
                 (369,229),
                 ],
                ]
            all_pts = [ [(x+radius,y+radius,subw,subh) for (x,y) in pts ] for pts in ll_pts ]
            for pts in all_pts:
                timestamp += 1
                us.add_frame( frame2, timestamp, pts )
            if close_it:
                us.close()

            us2 = ufmf.Ufmf(filename,seek_ok=seek_ok)
            assert isinstance(us2,ufmf.UfmfBase)
            test_frame1, test_timestamp1 = us2.get_bg_image()
            assert numpy.allclose( test_frame1, frame1 )
            assert test_timestamp1 == 1

            test_timestamp = 2
            for i, (timestamp, regions) in enumerate(us2.readframes()):
                progress = us2.get_progress()
                test_ll_pts = ll_pts[i]
                assert timestamp == test_timestamp
                test_timestamp += 1
                for test_ll,region in zip(test_ll_pts,regions):
                    xmin,ymin, bufim = region

                    # x
                    tj0 = test_ll[0]
                    tj1 = test_ll[0]+2*radius
                    # y
                    ti0 = test_ll[1]
                    ti1 = test_ll[1]+2*radius
                    if version==1:
                        # version 1 has fixed return size
                        if ti1 > frame2.shape[0]:
                            ti1 = frame2.shape[0]
                            ti0 = ti1-2*radius
                        if tj1 > frame2.shape[1]:
                            tj1 = frame2.shape[1]
                            tj0 = tj1-2*radius

                    testbuf = frame2[ti0:ti1,tj0:tj1]
                    assert xmin==tj0
                    assert ymin==ti0
                    assert testbuf.shape == bufim.shape
                    assert numpy.allclose( testbuf, bufim )
            us2.close()
        finally:
            os.unlink(filename)

def test_late_keyframe_fmf_emulator():
    w = 640
    h = 480

    frame1 = numpy.zeros( (h,w), dtype = numpy.uint8 )
    frame1[20:30,30:32] = 255
    frame1[20:30,30:32] = 255

    timestamp = 1
    radius = 15

    filename = tempfile.mkstemp()[1]
    subw = 2*radius
    subh = 2*radius
    try:
        us = ufmf.UfmfSaverV3( filename,
                               max_width=w,
                               max_height=h)
        assert isinstance(us,ufmf.UfmfSaverBase)
        frame2 = numpy.zeros( (h,w), dtype = numpy.uint8 )
        frame2[:,0::2] = range(0, w, 2) # clips (broadcast)
        for i in range(h):
            frame2[i,1::2] = i

        ll_pts = [
            [ (345, 144),
              (521, 118),
              ],
            [ (347, 144),
              (522, 119),
              (367, 229),
              ],
            [(349,145),
             (522,120),
             (369,229),
             ],
            ]
        all_pts = [ [(x+radius,y+radius,subw,subh) for (x,y) in pts ] for pts in ll_pts ]
        for pts in all_pts:
            timestamp += 1
            us.add_frame( frame2, timestamp, pts )
        # add keyframe after frame data
        us.add_keyframe( 'mean', frame1, timestamp+10 )
        us.close()

        fmf = ufmf.FlyMovieEmulator(filename)
        test_frame, test_timestamp = fmf.get_next_frame()
        fmf.close()
    finally:
        os.unlink(filename)
