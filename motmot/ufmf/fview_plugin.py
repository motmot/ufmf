from __future__ import with_statement, division
import pkg_resources

import motmot.fview.traited_plugin as traited_plugin
import motmot.ufmf.ufmf as ufmf

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group
import motmot.realtime_image_analysis.trealtime_image_analysis as tra
import motmot.FastImage.FastImage as FastImage
import threading

class CrossThreadEvent(traits.HasTraits):
    """like threading.Event class, with button to set the event"""
    _threading_event = traits.Any(transient=True)
    fire_event = traits.Button

    is_set = traits.Any(transient=True)
    set = traits.Any(transient=True)
    clear = traits.Any(transient=True)

    traits_view = View(Group(Item(name='fire_event',show_label=False)))

    def __init__(self,*args,**kwargs):
        super(CrossThreadEvent,self).__init__(*args,**kwargs)
        self._threading_event = threading.Event()
        self.is_set = self._threading_event.is_set
        self.set = self._threading_event.set
        self.clear = self._threading_event.clear

    def _fire_event_fired(self):
        self._threading_event.set()

class UFMFFviewSaver(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = 'ufmf saver'
    saving = traits.Bool(False)
    start_saving = traits.Button()
    stop_saving = traits.Button()
    clear_and_take_BG = traits.Instance(CrossThreadEvent,transient=True)
    bg_Nth_frame = traits.Int(100)
    use_roi2 = traits.Bool(True)
    realtime_analyzer = traits.Instance( tra.TraitedRealtimeAnalyzer )

    traits_view = View(Group(Item(name='start_saving',show_label=False),
                             Item(name='stop_saving',show_label=False),
                             Item(name='bg_Nth_frame'),
                             Item(name='use_roi2'),
                             Item(name='clear_and_take_BG',style='custom'),
                             Item(name='realtime_analyzer',
                                  style='custom'),
                             )
                       )
    def __init__(self,*args,**kwargs):
        super(UFMFFviewSaver,self).__init__(*args,**kwargs)
        self.clear_and_take_BG = CrossThreadEvent()

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
        if self.clear_and_take_BG.is_set():
            print 'clear_and_take_BG'
            self.clear_and_take_BG.clear()
        self.realtime_analyzer.do_work( fibuf, timestamp, framenumber,
                                        self.use_roi2)
        return draw_points, draw_linesegs
