import motmot.fview.traited_plugin as traited_plugin

class UFMFFviewSaver(traited_plugin.HasTraits_FViewPlugin):
    plugin_name = 'ufmf saver'

    def camera_starting_notification(self,cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):

        # This function gets called from FView when a camera is
        # initialized.

        return

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        draw_points = []
        draw_linesegs = []

        # This function gets called from FView immediately after
        # acquisition of each frame. Implement your image processing
        # logic here.

        return draw_points, draw_linesegs
