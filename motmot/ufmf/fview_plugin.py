from __future__ import with_statement, division
import pkg_resources

import motmot.fview.traited_plugin as traited_plugin
import motmot.ufmf.ufmf as ufmf

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.realtime_image_analysis.trealtime_image_analysis as tra
import motmot.FastImage.FastImage as FastImage
import threading

class UFMFFviewSaver(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = 'ufmf saver'
    saving = traits.Bool(False)
    start_saving = traits.Button()
    stop_saving = traits.Button()
    clear_and_take_BG = traits.Button()
    _clear_take_event = traits.Any(transient=True)
    bg_Nth_frame = traits.Int(100)
    use_roi2 = traits.Bool(True)
    realtime_analyzer = traits.Instance( tra.TraitedRealtimeAnalyzer )

    traits_view = View(Group(Item(name='start_saving',show_label=False),
                             Item(name='stop_saving',show_label=False),
                             Item(name='bg_Nth_frame'),
                             Item(name='use_roi2'),
                             Item(name='clear_and_take_BG'),
                             Item(name='realtime_analyzer',
                                  style='custom'),
                             )
                       )
    def __init__(self,*args,**kwargs):
        super(UFMFFviewSaver,self).__init__(*args,**kwargs)
        self._clear_take_event = threading.Event()

    def _clear_and_take_BG_fired(self):
        self._clear_take_event.set()

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

        fibuf = FastImage.asfastimage(buf)
        l,b = buf_offset
        lbrt = l, b, l+fibuf.size.w-1, b+fibuf.size.h-1
        if self._clear_take_event.is_set():
            self._clear_take_event.clear()
        self.realtime_analyzer.do_work( fibuf, timestamp, framenumber,
                                        self.use_roi2)
        return draw_points, draw_linesegs
