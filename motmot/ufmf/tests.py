import unittest
import ufmf
import pkg_resources # requires setuptools
import numpy

class TestFMF(unittest.TestCase):

    def test_a(self):
        w = 640
        h = 480
        
        frame1 = numpy.zeros( (h,w), dtype = numpy.uint8 )
        frame1[20:30,30:32] = 255
        frame1[20:30,30:32] = 255

        timestamp1 = 1e-20

        filename = 'testufmf.ufmf'
        us = ufmf.UfmfSaver( filename,
                             frame1,
                             timestamp1,
                             image_radius=5 )


        frame2 = numpy.zeros( (h,w), dtype = numpy.uint8 )

        
        frame2[20:30,30:40] = 1
        frame2[21,31] = 8
        frame2[60:70, 80:90] = 100
        frame2[61,81] = 7
        timestamp2 = 1

        pts = [ (35,25), # x,y
                (85,65),
                ]

        us.add_frame( frame2, timestamp2, pts )
        
        us.close()

        us2 = ufmf.Ufmf(filename)
        test_frame1 = us2.get_full_image()
        assert numpy.allclose( test_frame1, frame1 )
        print 'frame1 OK'
        us2._dump_frames()
        us2.close()

def get_test_suite():
    ts=unittest.TestSuite([unittest.makeSuite(TestFMF),
                           ])
    return ts

if __name__=='__main__':
    unittest.main()
