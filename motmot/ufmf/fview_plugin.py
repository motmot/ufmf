from __future__ import with_statement, division
import pkg_resources

import motmot.fview.traited_plugin as traited_plugin

import enthought.traits.api as traits
from enthought.traits.ui.api import View, Item, Group

class UFMFFviewSaver(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = 'ufmf saver'
    enabled = traits.Bool(False)
    draw_points = traits.Bool(True)
    draw_boxes = traits.Bool(True)
    pixel_format = traits.String()

    traits_view = View(Group(Item(name='enabled'),

                             Item(name='draw_points'),
                             Item(name='draw_boxes'),

                             Item(name='realtime_analyzer',
                                  style='custom'),
                             )
                       )

    def quit(self):
        if self.realtime_analyzer is not None:
            self.realtime_analyzer.quit()
            self.realtime_analyzer = None

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        self.realtime_analyzer = None

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        draw_points = []
        draw_linesegs = []

        if self.enabled:
            xpoints, actual_saved_points = self.realtime_analyzer.process_frame(
                buf,buf_offset,timestamp,framenumber)

            if self.draw_points:
                for pt in xpoints:
                    draw_points.append( pt[:2])

            if self.draw_boxes:
                for pt in actual_saved_points:
                    x0,y0, x1,y1 = pt
                    draw_linesegs.extend( [(x0,y0, x0,y1),
                                           (x0,y1, x1,y1),
                                           (x1,y1, x1,y0),
                                           (x1,y0, x0,y0)])
        return draw_points, draw_linesegs
