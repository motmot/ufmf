import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.ufmf.ufmf as ufmf_mod

import motmot.realtime_image_analysis.trealtime_image_analysis as tra
import motmot.FastImage.FastImage as FastImage

class UfmfWriter(traits.HasTraits):
    """write a .ufmf file from series of input frames.

    This is class performs image analysis to save only the parts of an
    image that change. For a bare-bones saver, see the UfmfSaver class
    in ufmf.py.
    """
    max_width = traits.Int(None,transient=True) # XXX how to make set-once?
    max_height = traits.Int(None,transient=True) # XXX how to make set-once?
    filename = traits.String(None,transient=True)

    difference_threshold = traits.Float(5.0)
    bg_Nth_frame = traits.Int(100)
    use_roi2 = traits.Bool(True)
    realtime_analyzer = traits.Instance( tra.TraitedRealtimeAnalyzer,
                                         transient=True )

    # default (basic) view. (FIXME TODO: Add advanced view with all options.)
    traits_view = View(Group(Item(name='bg_Nth_frame'),
                             Item(name='difference_threshold'),
                             Item(name='use_roi2'),
                             Item(name='realtime_analyzer',
                                  style='custom'),
                             )
                       )
    def __init__(self,*args,**kwargs):
        super(UfmfWriter,self).__init__(*args,**kwargs)
        if self.filename is None:
            raise ValueError('filename must be specified')
        if self.max_width is None:
            raise ValueError('max_width must be specified')
        if self.max_height is None:
            raise ValueError('max_height must be specified')
        self.realtime_analyzer = tra.TraitedRealtimeAnalyzer(
            max_width=self.max_width,
            max_height=self.max_height)

        self.max_frame_size = FastImage.Size( self.max_width, self.max_height )
        self.full_frame_live = FastImage.FastImage8u( self.max_frame_size )
        self.running_mean_im = FastImage.FastImage32f( self.max_frame_size)
        self.ufmf = ufmf_mod.UfmfSaver(self.filename)

    def add_frame(self, frame, timestamp, frame_offset=None ):
        """add a new frame to the .ufmf"""
        if frame_offset is None:
            frame_offset = (0,0)

    def close(self):
        pass
