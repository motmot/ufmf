from __future__ import division
from __future__ import with_statement

import sys, threading, Queue, time, socket, math, struct, os
import pkg_resources

import motmot.imops.imops as imops
import motmot.FastImage.FastImage as FastImage

import motmot.realtime_image_analysis.realtime_image_analysis as realtime_image_analysis

import motmot.ufmf.ufmf as ufmf

import numpy

import motmot.wxvalidatedtext.wxvalidatedtext as wxvt

import warnings

import wx
from wx import xrc
import scipy.io

RESFILE = pkg_resources.resource_filename(__name__,"ufmf_flytrax.xrc") # trigger extraction
RES = xrc.EmptyXmlResource()
RES.LoadFromString(open(RESFILE).read())

class BunchClass(object):
    pass

class BufferAllocator:
    def __call__(self, w, h):
        return FastImage.FastImage8u(FastImage.Size(w,h))

class SharedValue:
    def __init__(self):
        self.evt = threading.Event()
        self._val = None
    def set(self,value):
        # called from producer thread
        self._val = value
        self.evt.set()
    def is_new_value_waiting(self):
        return self.evt.isSet()
    def get(self,*args,**kwargs):
        # called from consumer thread
        self.evt.wait(*args,**kwargs)
        val = self._val
        self.evt.clear()
        return val
    def get_nowait(self):
        val = self._val
        self.evt.clear()
        return val

class LockedValue:
    def __init__(self,initial_value=None):
        self._val = initial_value
        self._q = Queue.Queue()
    def set(self,value):
        self._q.put( value )
    def get(self):
        try:
            while 1:
                self._val = self._q.get_nowait()
        except Queue.Empty:
            pass
        return self._val

def corners2linesegs( xmin, ymin, xmax, ymax ):
    return [ [xmin,ymin,xmin,ymax],
             [xmin,ymax,xmax,ymax],
             [xmax,ymax,xmax,ymin],
             [xmax,ymin,xmin,ymin],
             ]

class Tracker(object):
    def __init__(self,wx_parent):
        self.wx_parent = wx_parent
        self.frame = RES.LoadFrame(self.wx_parent,"UFMF_FLYTRAX_FRAME") # make frame main panel

        self.last_n_downstream_hosts = None

        self.frame_nb = xrc.XRCCTRL(self.frame,"FLYTRAX_NOTEBOOK")
        self.status_message = xrc.XRCCTRL(self.frame,"STATUS_MESSAGE")
        self.status_message2 = xrc.XRCCTRL(self.frame,"STATUS_MESSAGE2")
        self.new_image = False

        self.cam_ids = []
        self.pixel_format = {}
        self.bunches = {}

        self.ufmf_writer = {}

        self.clear_and_take_bg_image = {}

        self.ongoing_bg_image_update_interval = {}

        self.realtime_analyzer = {}
        self.max_frame_size = {}

        self.clear_threshold_value = {}
        self.new_clear_threshold = {}
        self.diff_threshold_value = {}
        self.new_diff_threshold = {}
        self.roi2_radius_value = {}
        self.new_roi2_radius = {}
        self.display_active = {}

        self.save_status_widget = {}
        self.save_data_prefix_widget = {}

        self.widget2cam_id = {}

        self.image_update_lock = threading.Lock()

        self.bg_update_lock = threading.Lock()

        self.sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.minimum_eccentricity = 1.5

        self.per_cam_panel = {}

        self.ticks_since_last_update = {}

        ID_Timer = wx.NewId()
        self.timer_clear_message = wx.Timer(self.wx_parent, ID_Timer)
        wx.EVT_TIMER(self.wx_parent, ID_Timer, self.OnClearMessage)

        self.full_bg_image = {}
        self.xrcid2validator = {}

    def get_frame(self):
        return self.frame

    def camera_starting_notification(self,
                                     cam_id,
                                     pixel_format=None,
                                     max_width=None,
                                     max_height=None):
        """

        cam_id is simply used as a dict key

        """
        self.xrcid2validator[cam_id] = {}

        bunch = BunchClass()

        bunch.max_num_points = SharedValue()
        bunch.max_num_points.set(10)

        self.bunches[cam_id]=bunch
        self.pixel_format[cam_id]=pixel_format
        # setup GUI stuff
        if len(self.cam_ids)==0:
            # adding first camera
            self.frame_nb.DeleteAllPages()

        #  make new per-camera wx panel
        per_cam_panel = RES.LoadPanel(self.frame_nb,"PER_CAM_PANEL")
        self.per_cam_panel[cam_id] = per_cam_panel
        per_cam_panel.SetAutoLayout(True)
        self.frame_nb.AddPage(per_cam_panel,cam_id)

        ctrl = xrc.XRCCTRL(per_cam_panel,"TAKE_BG_IMAGE")
        self.widget2cam_id[ctrl]=cam_id
        wx.EVT_BUTTON(ctrl,
                      ctrl.GetId(),
                      self.OnTakeBgImage)

        self.ongoing_bg_image_update_interval[cam_id] = LockedValue(50)
        ctrl = xrc.XRCCTRL(per_cam_panel,"BACKGROUND_IMAGE_UPDATE_INTERVAL")
        ctrl.SetValue( str(self.ongoing_bg_image_update_interval[cam_id].get()))
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_integer_callback(
            ctrl,
            ctrl.GetId(),
            self.OnSetBackgroundUpdateInterval)
        self.xrcid2validator[cam_id]["BACKGROUND_IMAGE_UPDATE_INTERVAL"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"CLEAR_THRESHOLD")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_float_callback(
            ctrl,
            ctrl.GetId(),
            self.OnClearThreshold,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["CLEAR_THRESHOLD"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"DIFF_THRESHOLD")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_integer_callback(
            ctrl,
            ctrl.GetId(),
            self.OnDiffThreshold,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["DIFF_THRESHOLD"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"ROI2_RADIUS")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_integer_callback(
            ctrl,
            ctrl.GetId(),
            self.OnRoi2Radius,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["ROI2_RADIUS"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"INVERSE_ALPHA")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_float_callback(
            ctrl,
            ctrl.GetId(),
            self.OnInverseAlpha,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["INVERSE_ALPHA"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"N_SIGMA")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_float_callback(
            ctrl,
            ctrl.GetId(),
            self.OnNSigma,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["N_SIGMA"] = validator

        ctrl = xrc.XRCCTRL(per_cam_panel,"MAX_NUM_POINTS")
        self.widget2cam_id[ctrl]=cam_id
        validator = wxvt.setup_validated_integer_callback(
            ctrl,
            ctrl.GetId(),
            self.OnMaxNPoints,
            ignore_initial_value=True)
        self.xrcid2validator[cam_id]["MAX_NUM_POINTS"] = validator
        ctrl.SetValue(str(bunch.max_num_points.get_nowait()))
        validator.set_state('valid')

        start_recording_widget = xrc.XRCCTRL(per_cam_panel,"START_RECORDING")
        self.widget2cam_id[start_recording_widget]=cam_id
        wx.EVT_BUTTON(start_recording_widget,
                      start_recording_widget.GetId(),
                      self.OnStartRecording)

        stop_recording_widget = xrc.XRCCTRL(per_cam_panel,"STOP_RECORDING")
        self.widget2cam_id[stop_recording_widget]=cam_id
        wx.EVT_BUTTON(stop_recording_widget,
                      stop_recording_widget.GetId(),
                      self.OnStopRecording)

        save_status_widget = xrc.XRCCTRL(per_cam_panel,"SAVE_STATUS")
        self.save_status_widget[cam_id] = save_status_widget

        self.save_data_prefix_widget[cam_id] = xrc.XRCCTRL(
            per_cam_panel,"SAVE_DATA_PREFIX")
        self.widget2cam_id[self.save_data_prefix_widget[cam_id]]=cam_id

        ctrl = xrc.XRCCTRL(per_cam_panel,"SAVE_DATA_DIR_BUTTON")
        self.widget2cam_id[ctrl]=cam_id
        wx.EVT_BUTTON(ctrl,
                      ctrl.GetId(),
                      self.OnSetSavePath)

#####################

        # setup non-GUI stuff
        self.cam_ids.append(cam_id)

        self.display_active[cam_id] = threading.Event()
        if len(self.cam_ids) > 1:
            raise NotImplementedError('if >1 camera supported, implement setting display_active on notebook page change')
        else:
            self.display_active[cam_id].set()

        self.clear_and_take_bg_image[cam_id] = threading.Event()
        self.clear_and_take_bg_image[cam_id].set() # start off setting background

        self.ticks_since_last_update[cam_id] = 0
        lbrt = (0,0,max_width-1,max_height-1)
        
        roi2_radius=int(xrc.XRCCTRL(per_cam_panel,"ROI2_RADIUS").GetValue())
        ra = realtime_image_analysis.RealtimeAnalyzer(lbrt,
                                                      max_width,
                                                      max_height,
                                                      bunch.max_num_points.get_nowait(),
                                                      roi2_radius)
        self.realtime_analyzer[cam_id] = ra

        self.new_clear_threshold[cam_id] = threading.Event()
        self.new_diff_threshold[cam_id] = threading.Event()
        self.new_roi2_radius[cam_id] = threading.Event()

        self.clear_threshold_value[cam_id] = ra.clear_threshold
        self.diff_threshold_value[cam_id] = ra.diff_threshold
        self.roi2_radius_value[cam_id] = ra.roi2_radius

        ctrl = xrc.XRCCTRL(per_cam_panel,"CLEAR_THRESHOLD")
        validator = self.xrcid2validator[cam_id]["CLEAR_THRESHOLD"]
        ctrl.SetValue( '%.2f'%ra.clear_threshold )
        validator.set_state('valid')

        ctrl = xrc.XRCCTRL(per_cam_panel,"DIFF_THRESHOLD")
        validator = self.xrcid2validator[cam_id]["DIFF_THRESHOLD"]
        ctrl.SetValue( '%d'%ra.diff_threshold )
        validator.set_state('valid')

        ctrl = xrc.XRCCTRL(per_cam_panel,"ROI2_RADIUS")
        validator = self.xrcid2validator[cam_id]["ROI2_RADIUS"]
        ctrl.SetValue( '%d'%ra.roi2_radius )
        validator.set_state('valid')

        max_frame_size = FastImage.Size( max_width, max_height )
        self.max_frame_size[cam_id] = max_frame_size

        bunch.inverse_alpha = SharedValue()
        bunch.inverse_alpha.set( float(xrc.XRCCTRL(per_cam_panel,"INVERSE_ALPHA").GetValue()) )
        validator = self.xrcid2validator[cam_id]["INVERSE_ALPHA"]
        validator.set_state('valid')

        bunch.n_sigma = SharedValue()
        bunch.n_sigma.set( float(xrc.XRCCTRL(per_cam_panel,"N_SIGMA").GetValue()) )
        validator = self.xrcid2validator[cam_id]["N_SIGMA"]
        validator.set_state('valid')

        bunch.initial_take_bg_state = None

        bunch.running_mean_im_full = FastImage.FastImage32f(max_frame_size)
        bunch.running_sumsqf_full = FastImage.FastImage32f(max_frame_size)
        bunch.running_sumsqf_full.set_val(0,max_frame_size)
        bunch.fastframef32_tmp_full = FastImage.FastImage32f(max_frame_size)
        bunch.mean2_full = FastImage.FastImage32f(max_frame_size)
        bunch.std2_full = FastImage.FastImage32f(max_frame_size)
        bunch.running_stdframe_full = FastImage.FastImage32f(max_frame_size)
        bunch.noisy_pixels_mask_full = FastImage.FastImage8u(max_frame_size)

        bunch.last_running_mean_im = None

    def get_buffer_allocator(self,cam_id):
        return BufferAllocator()

    def get_plugin_name(self):
        return 'UFMF FlyTrax'

    def OnTakeBgImage(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]

        per_cam_panel = self.per_cam_panel[cam_id]
        ctrl = xrc.XRCCTRL(per_cam_panel,"TAKE_BG_IMAGE_ALLOW_WHEN_SAVING")
        if not ctrl.GetValue() and cam_id in self.ufmf_writer:

            dlg = wx.MessageDialog(self.wx_parent,
                                   'Saving data - cannot take background image',
                                   'UFMF FlyTrax error',
                                   wx.OK | wx.ICON_ERROR
                                   )
            dlg.ShowModal()
            dlg.Destroy()
            return

        self.clear_and_take_bg_image[cam_id].set()
        self.display_message('capturing background image')

    def OnSetBackgroundUpdateInterval(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        val = int(widget.GetValue())
        self.ongoing_bg_image_update_interval[cam_id].set(val)

    def OnClearThreshold(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        newvalstr = widget.GetValue()
        try:
            newval = float(newvalstr)
        except ValueError:
            pass
        else:
            # only touch realtime_analysis in other thread
            self.clear_threshold_value[cam_id] = newval
            self.new_clear_threshold[cam_id].set()
            self.display_message('set clear threshold %s'%str(newval))
        event.Skip()

    def OnDiffThreshold(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        newvalstr = widget.GetValue()
        try:
            newval = int(newvalstr)
        except ValueError:
            pass
        else:
            # only touch realtime_analysis in other thread
            self.diff_threshold_value[cam_id] = newval
            self.new_diff_threshold[cam_id].set()
            self.display_message('set difference threshold %d'%newval)
        event.Skip()

    def OnRoi2Radius(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        newvalstr = widget.GetValue()
        try:
            newval = int(newvalstr)
        except ValueError:
            pass
        else:
            # only touch realtime_analysis in other thread
            self.roi2_radius_value[cam_id] = newval
            self.new_roi2_radius[cam_id].set()
            self.display_message('set UFMF window size %d'%newval)
        event.Skip()

    def OnInverseAlpha(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        bunch = self.bunches[cam_id]

        newvalstr = widget.GetValue()
        try:
            newval = float(newvalstr)
        except ValueError:
            pass
        else:
            bunch.inverse_alpha.set( newval )
        event.Skip()

    def OnNSigma(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        bunch = self.bunches[cam_id]

        newvalstr = widget.GetValue()
        try:
            newval = float(newvalstr)
        except ValueError:
            pass
        else:
            bunch.n_sigma.set( newval )
        event.Skip()

    def OnMaxNPoints(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]
        bunch = self.bunches[cam_id]

        newvalstr = widget.GetValue()
        try:
            newval = int(newvalstr)
        except ValueError:
            pass
        else:
            bunch.max_num_points.set( newval )
        event.Skip()

    def process_frame(self,cam_id,buf,buf_offset,timestamp,framenumber):
        if self.pixel_format[cam_id]=='YUV422':
            buf = imops.yuv422_to_mono8( numpy.asarray(buf) ) # convert
        elif not self.pixel_format[cam_id].startswith('MONO8'):
            warnings.warn("flytrax plugin incompatible with data format")
            return [], []


        bunch = self.bunches[cam_id]

        do_bg_maint = False
        clear_and_take_bg_image = self.clear_and_take_bg_image[cam_id]

        # this is called in realtime thread
        fibuf = FastImage.asfastimage(buf) # FastImage view of image data (hardware ROI)
        l,b = buf_offset
        lbrt = l, b, l+fibuf.size.w-1, b+fibuf.size.h-1

        running_mean_im = bunch.running_mean_im_full.roi(l, b, fibuf.size)  # set ROI view
        running_sumsqf = bunch.running_sumsqf_full.roi(l, b, fibuf.size)  # set ROI view

        new_clear_threshold = self.new_clear_threshold[cam_id]
        new_diff_threshold = self.new_diff_threshold[cam_id]
        new_roi2_radius = self.new_roi2_radius[cam_id]
        realtime_analyzer = self.realtime_analyzer[cam_id]
        realtime_analyzer.roi = lbrt # hardware ROI
        max_frame_size = self.max_frame_size[cam_id]
        display_active = self.display_active[cam_id]

        use_roi2 = True

        use_cmp = False # use variance-based background subtraction/analysis
        draw_points = []
        draw_linesegs = []

        running_mean8u_im_full = realtime_analyzer.get_image_view('mean')
        running_mean8u_im = running_mean8u_im_full.roi(l, b, fibuf.size)

        if (bunch.initial_take_bg_state is not None or 
            clear_and_take_bg_image.isSet()):
            src_fullframe_fi = fibuf.get_8u_copy(max_frame_size)

        if bunch.initial_take_bg_state is not None:
            assert bunch.initial_take_bg_state == 'gather'

            n_initial_take = 5
            bunch.initial_take_frames.append( numpy.array(src_fullframe_fi) ) # copied above
            if len( bunch.initial_take_frames ) >= n_initial_take:

                initial_take_frames = numpy.array( bunch.initial_take_frames, dtype=numpy.float32 )
                mean_frame = numpy.mean( initial_take_frames, axis=0)
                sumsqf_frame = numpy.sum(initial_take_frames**2, axis=0)/len( initial_take_frames )

                numpy.asarray(running_mean_im)[:,:] = mean_frame
                numpy.asarray(running_sumsqf)[:,:] = sumsqf_frame

                # we're done with initial transient, set stuff
                do_bg_maint = True
                bunch.initial_take_bg_state = None
                bunch.initial_take_frames = []

        if clear_and_take_bg_image.isSet():
            bunch.initial_take_bg_state = 'gather'
            bunch.initial_take_frames = []
            with self.bg_update_lock:
                bunch.last_running_mean_im = None
            clear_and_take_bg_image.clear()

        if 1:
            self.ticks_since_last_update[cam_id] += 1

            update_interval = self.ongoing_bg_image_update_interval[cam_id].get()
            if self.ticks_since_last_update[cam_id]%update_interval == 0:
                do_bg_maint = True

        ufmf_writer = self.ufmf_writer.get(cam_id,None)

        if do_bg_maint:
            hw_roi_frame = fibuf
            cur_fisize = hw_roi_frame.size
            bg_frame_alpha = 1.0/bunch.inverse_alpha.get_nowait()
            n_sigma = bunch.n_sigma.get_nowait()
            bright_non_gaussian_cutoff = 255
            bright_non_gaussian_replacement = 255

            compareframe8u_full = realtime_analyzer.get_image_view('cmp')
            compareframe8u = compareframe8u_full.roi(l, b, fibuf.size)
            fastframef32_tmp = bunch.fastframef32_tmp_full.roi(l, b, fibuf.size)

            mean2 = bunch.mean2_full.roi(l, b, fibuf.size)
            std2  =  bunch.std2_full.roi(l, b, fibuf.size)
            running_stdframe = bunch.running_stdframe_full.roi(l, b, fibuf.size)

            noisy_pixels_mask = bunch.noisy_pixels_mask_full.roi(l, b, fibuf.size)

            realtime_image_analysis.do_bg_maint(
                running_mean_im,#in
                hw_roi_frame,#in
                cur_fisize,#in
                bg_frame_alpha, #in
                running_mean8u_im,
                fastframef32_tmp,
                running_sumsqf, #in
                mean2,
                std2,
                running_stdframe,
                n_sigma,#in
                compareframe8u,
                bright_non_gaussian_cutoff,#in
                noisy_pixels_mask,#in
                bright_non_gaussian_replacement,#in
                bench=0 )
                #debug=0)
            #chainbuf.real_std_est= tmpresult
            bg_changed = True
            bg_frame_number = 0

            with self.bg_update_lock:
                bunch.last_running_mean_im = running_mean_im
                bunch.last_running_sumsqf_image = running_sumsqf
                bunch.last_bgcmp_image_timestamp = timestamp

            if ufmf_writer is not None:
                ufmf_writer.add_keyframe('mean',
                                         running_mean_im,
                                         timestamp)
                ufmf_writer.add_keyframe('sumsq',
                                         running_sumsqf,
                                         timestamp)

        if new_clear_threshold.isSet():
            nv = self.clear_threshold_value[cam_id]
            realtime_analyzer.clear_threshold = nv
            #print 'set clear',nv
            new_clear_threshold.clear()

        if new_diff_threshold.isSet():
            nv = self.diff_threshold_value[cam_id]
            realtime_analyzer.diff_threshold = nv
            #print 'set diff',nv
            new_diff_threshold.clear()

        if new_roi2_radius.isSet():
            nv = self.roi2_radius_value[cam_id]
            realtime_analyzer.roi2_radius = nv
            new_roi2_radius.clear()

        n_pts = 0
        if ufmf_writer is not None:
            try:
                realtime_analyzer.max_num_points = bunch.max_num_points.get_nowait()
            except AttributeError, err:
                warnings.warn('old realtime_analyzer does not support dynamic setting of max_num_points')
            points = realtime_analyzer.do_work(fibuf,
                                               timestamp, framenumber, use_roi2,
                                               use_cmp=use_cmp)
            pts = []
            w = h = realtime_analyzer.roi2_radius*2
            for pt in points:
                pts.append( (pt[0], pt[1], w, h ) )
            saved_points = ufmf_writer.add_frame( fibuf, timestamp, pts )
            lineseg_lists = [ corners2linesegs( *corners ) for corners in saved_points]
            for linesegs in lineseg_lists:
                draw_linesegs.extend( linesegs )

        return draw_points, draw_linesegs

    def display_message(self,msg,duration_msec=2000):
        self.status_message.SetLabel(msg)
        self.timer_clear_message.Start(duration_msec,wx.TIMER_ONE_SHOT)

    def OnClearMessage(self,evt):
        self.status_message.SetLabel('')

    def OnSetSavePath(self, event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]

        prefix = self.save_data_prefix_widget[cam_id].GetValue()
        defaultPath, prefix_fname = os.path.split(prefix)

        dlg = wx.DirDialog( self.frame, "ufmf record directory",
                            style = wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
                            defaultPath = defaultPath,
                            )
        try:
            if dlg.ShowModal() == wx.ID_OK:
                save_path_dir = dlg.GetPath()
                prefix = os.path.join( save_path_dir, prefix_fname )
                self.save_data_prefix_widget[cam_id].SetValue(prefix)
        finally:
            dlg.Destroy()

    def OnStartRecording(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]

        if cam_id in self.ufmf_writer:
            self.display_message("already saving data: not starting")
            return

        per_cam_panel = self.per_cam_panel[cam_id]

        bunch = self.bunches[cam_id]

        # grab background image from other thread
        with self.bg_update_lock:
            last_running_mean_im = bunch.last_running_mean_im
            if last_running_mean_im is not None:
                last_bgcmp_image_timestamp = bunch.last_bgcmp_image_timestamp
                last_running_sumsqf_image = bunch.last_running_sumsqf_image

        prefix = self.save_data_prefix_widget[cam_id].GetValue()
        fname = prefix+time.strftime('%Y%m%d_%H%M%S.ufmf')
        ufmf_writer = ufmf.AutoShrinkUfmfSaverV3( fname,
                                                  coding = self.pixel_format[cam_id],
                                                  max_width=self.max_frame_size[cam_id].w,
                                                  max_height=self.max_frame_size[cam_id].h,
                                                  )
        bunch = self.bunches[cam_id]

        if last_running_mean_im is not None:
            ufmf_writer.add_keyframe('mean',
                                     last_running_mean_im,
                                     last_bgcmp_image_timestamp)
            ufmf_writer.add_keyframe('sumsq',
                                     last_running_sumsqf_image,
                                     last_bgcmp_image_timestamp)
        self.ufmf_writer[cam_id] = ufmf_writer
        self.save_status_widget[cam_id].SetLabel('saving')
        self.display_message('saving data to %s'%fname)

    def OnStopRecording(self,event):
        widget = event.GetEventObject()
        cam_id = self.widget2cam_id[widget]

        if cam_id in self.ufmf_writer:
            self.ufmf_writer[cam_id].close()
            del self.ufmf_writer[cam_id]
            self.save_status_widget[cam_id].SetLabel('not saving')

            per_cam_panel = self.per_cam_panel[cam_id]
        else:
            self.display_message("not saving data: not stopping")

    def quit(self):
        for ufmf_writer in self.ufmf_writer.itervalues():
            ufmf_writer.close() # make sure all data savers close nicely
