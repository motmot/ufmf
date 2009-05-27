from __future__ import with_statement, division
import pkg_resources

import motmot.fview.traited_plugin as traited_plugin
import motmot.ufmf.ufmf as ufmf

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.realtime_image_analysis.trealtime_image_analysis as tra
import motmot.FastImage.FastImage as FastImage

class UFMFFviewSaver(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = 'ufmf saver'
    saving = traits.Bool(False)
    start_saving = traits.Button()
    stop_saving = traits.Button()
    bg_Nth_frame = traits.Int(100)
    use_roi2 = traits.Bool(True)
    realtime_analyzer = traits.Instance( tra.TraitedRealtimeAnalyzer )

    traits_view = View(Group(Item(name='start_saving',show_label=False),
                             Item(name='stop_saving',show_label=False),
                             Item(name='bg_Nth_frame'),
                             Item(name='use_roi2'),
                             Item(name='realtime_analyzer',
                                  style='custom'),
                             )
                       )

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        self.realtime_analyzer = tra.TraitedRealtimeAnalyzer(
            max_width=max_width,
            max_height=max_height)

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        draw_points = []
        draw_linesegs = []

        # This function gets called from FView immediately after
        # acquisition of each frame. Implement your image processing
        # logic here.

        raw_im_small = FastImage.asfastimage(buf)
        self.realtime_analyzer.do_work( raw_im_small, timestamp, framenumber,
                                        self.use_roi2)
        return draw_points, draw_linesegs
