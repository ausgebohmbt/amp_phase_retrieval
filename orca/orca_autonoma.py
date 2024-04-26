import orca.hamamatsu_camera as hc
import abc
import numpy as np
import time


class Camera():
    """Base class for all cameras"""
    __metaclass__ = abc.ABCMeta
    # def signals to trigger main

    def __init__(self):
        # super(Camera, self).__init__(parent)
        self.num = 1
        self.last_frame = np.zeros((2048, 2048))
        self.integration_mode = False

        self.bckgr = np.zeros((2048, 2048))

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def end(self):
        pass

    @abc.abstractmethod
    def terminate(self) -> None:
        super(Camera, self).terminate()

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def framerate(self, exposure):
        self._exposure = exposure

    @abc.abstractmethod
    def take_image(self) -> np.ndarray:
        """Capture and return a single image from the camera.

        Returns
        -------
        np.ndarray
            The captured image as a numpy array normalized by the maximum count.
        """

    @abc.abstractmethod
    def take_average_image(self, num: int):
        """Capture and return an average image from a number of captures.

        Parameters
        ----------
        num : int
            The number of images to capture and average.

        Returns
        -------
        np.ndarray
            The averaged image as a NumPy array.
        """
        print("getting average of {} images".format(num))
        images = np.stack([self.take_image() for _ in range(num)])
        # def acqr():
        #     self.mode = "Acq"
        #     self.start()
        #     self.wait()
        #     print("last frame max {}".format(np.max(self.last_frame)))
        #     return self.last_frame
        #
        # images = np.stack([acqr() for _ in range(num)])
        return images.mean(axis=0)

    def show_image(self, lognorm=False) -> None:
        pass

    # NOTE not all cameras might have a ROI feature, need to redesign if that's the case
    def center_to_max(self, shape: tuple[int, int]) -> None:
        pass


class BaseHamamatsu(Camera):
    """Base class for Hamamatsu cameras"""
    __metaclass__ = abc.ABCMeta

    def __init__(self, parent=None, **kwargs):
        super(Camera, self).__init__()
        if 'exposure' not in kwargs.keys():
            raise KeyError('No exposure value specified')
        print("initializing camera base")
        self.initCamState = kwargs["initCam"]
        cam_no = kwargs["came_numb"]
        self.trigg_mode = kwargs["trig_mODe"]
        self.delay_ms = 49.5/1000.0
        self.bin_sz = 1
        self.hcam = hc.HamamatsuCameraMR(camera_numb=cam_no, initCam=self.initCamState, trig_mODe=self.trigg_mode)
        self.sensor_temp = 42

        self.hcam.getPropertyValue("trigger_mode")
        self.hcam.getPropertyValue("trigger_source")
        self.hcam.getPropertyValue("sensor_cooler_status")
        self.hcam.getPropertyValue("sensor_temperature")

        [self.cam_x, _] = self.hcam.getPropertyValue("image_detector_pixel_num_horz")
        [self.cam_y, _] = self.hcam.getPropertyValue("image_detector_pixel_num_vert")

        self.hcam.setPropertyValue("defect_correct_mode", "OFF")
        self.hcam.setPropertyValue("subarray_hsize", self.cam_x)
        self.hcam.setPropertyValue("subarray_vsize", self.cam_y)
        self.hcam.setPropertyValue("binning", "{}x{}".format(self.bin_sz, self.bin_sz))
        self.hcam.setPropertyValue("readout_speed", 2)  # slow readout mode is 2, 30fps max
        self.hcam.getPropertyValue("sensor_mode")
        self.hcam.getPropertyValue("readout_speed")
        self.hcam.getPropertyValue("binning")

        self.exposure = kwargs['exposure']
        self.roi_is = [0, 0, 2048, 2048]  # FIXME: fixmeeeeeeeeee
        self.img_sz_y = int(np.floor(self.cam_y / self.bin_sz))
        self.img_sz_x = int(np.floor(self.cam_x / self.bin_sz))
        self.last_frame = np.zeros((self.img_sz_y, self.img_sz_x))
        self.res = (2048, 2048)
        self.roi = [0, 0, 2048, 2048]

    # todo: mk property/Up-setter
    def bin_value(self, new_bin):
        # fixme raises exception in Preview if in live mode and restart
        self.hcam.stopAcquisition()
        self.bin_sz = new_bin
        self.hcam.setPropertyValue("binning", "{}x{}".format(self.bin_sz, self.bin_sz))
        self.img_sz_y = int(np.floor(self.cam_y / self.bin_sz))
        self.img_sz_x = int(np.floor(self.cam_x / self.bin_sz))
        self.hcam.getPropertyValue("binning")

        print('get')
        print(self.hcam.getPropertyValue("subarray_hpos"))
        print(self.hcam.getPropertyValue("subarray_vpos"))
        print(self.hcam.getPropertyValue("subarray_hsize"))
        print(self.hcam.getPropertyValue("subarray_vsize"))

    def sensor_temp_is(self):
        # print("gettin temp")
        self.tmp.emit(self.hcam.getPropertyValue("sensor_temperature"))

    def roi_current(self):
        # FIXME: currently not used
        """returns,,, current roi"""
        hpos = self.hcam.getPropertyValue("subarray_hpos")
        vpos = self.hcam.getPropertyValue("subarray_vpos")
        hsize = self.hcam.getPropertyValue("subarray_hsize")
        vsize = self.hcam.getPropertyValue("subarray_vsize")
        self.roi_is = [hpos, vpos, hsize, vsize]

    def roi_set_roi(self, hpos, vpos, hsize, vsize):
        """sets roi from rectangle,
            pathetically, regions must be dividends of 4"""
        "if you choose to reinstate the -fliplr- condition in acquisition "
        "you will need to reconfigure the hpos value, think it should be like abs(2048-hpos)"
        "manual: in sub-array readout, binning configuration is enabled."

        hpos = hpos   # * self.bin_sz
        vpos = vpos   #  * self.bin_sz
        hsize = hsize   #  * self.bin_sz
        vsize = vsize   #  * self.bin_sz
        print("roi in: hpos, vpos, hsize, vsize: {}, {}, {}, {}".format(hpos, vpos, hsize, vsize))

        self.hcam.stopAcquisition()
        if self.roi_is[0] != 0 or self.roi_is[1] != 0:
            print("here")
            hpos_the = hpos + self.roi_is[0]
            vpos_the = vpos + self.roi_is[1]
        else:
            print("there")
            hpos_the = hpos
            vpos_the = vpos

        # # if self.bin_sz == 1:
        # if hsize + hpos_the > 2048:
        #     print('here, {}'.format(hsize + hpos_the))
        #     hsize = 2048 - hpos_the
        # if vsize + vpos_the > 2048:
        #     print('there, {}'.format(vsize + vpos_the))
        #     vsize = 2048 - vpos_the

        print("roi pre mod: hpos, vpos, hsize, vsize: {}, {}, {}, {}".format(hpos_the, vpos_the, hsize, vsize))


        if hsize % 4 != 0:
            hsize = hsize - (hsize % 4) + 4
        if hpos_the % 4 != 0:
            hpos_the = hpos_the - (hpos_the % 4) + 4
        if vsize % 4 != 0:
            vsize = vsize - (vsize % 4) + 4
        if vpos_the % 4 != 0:
            vpos_the = vpos_the - (vpos_the % 4) + 4

        print("roi in mod: hpos, vpos, hsize, vsize: {}, {}, {}, {}".format(hpos_the, vpos_the, hsize, vsize))
        # print("the hpos, vpos, hsize, vsize, {}, {}, {}, {}".format(hpos_the, vpos_the, hsize, vsize))

        self.hcam.setPropertyValue("subarray_hpos", hpos_the)
        self.hcam.setPropertyValue("subarray_vpos", vpos_the)
        self.hcam.setPropertyValue("subarray_hsize", hsize)
        self.hcam.setPropertyValue("subarray_vsize", vsize)
        self.hcam.setPropertyValue("subarray_mode", "ON")  # this is set in the low-level script in "setSubArrayMode"
        self.hcam.getPropertyValue("subarray_mode")  # this is set in the low-level script in "setSubArrayMode"

        # set the current values to self
        self.roi_is = [hpos_the, vpos_the, hsize, vsize]
        # read from cam
        new_hpos_the = self.hcam.getPropertyValue("subarray_hpos")
        new_vpos_the = self.hcam.getPropertyValue("subarray_vpos")
        new_hsize = self.hcam.getPropertyValue("subarray_hsize")
        new_vsize = self.hcam.getPropertyValue("subarray_vsize")
        print("new roi: hpos, vpos, hsize, vsize: {}, {}, {}, {}".format(new_hpos_the, new_vpos_the,
                                                                         new_hsize, new_vsize))
        self.img_sz_x = int(self.hcam.getPropertyValue("subarray_hsize")[0] / self.bin_sz)
        self.img_sz_y = int(self.hcam.getPropertyValue("subarray_vsize")[0] / self.bin_sz)
        # self.img_sz_x = int(self.hcam.getPropertyValue("subarray_hsize")[0])
        # self.img_sz_y = int(self.hcam.getPropertyValue("subarray_vsize")[0])
        # print('get')
        # print(self.hcam.getPropertyValue("subarray_hpos"))
        # print(self.hcam.getPropertyValue("subarray_vpos"))
        # print(self.img_sz_x)
        # print(self.img_sz_y)
        # print(self.hcam.getPropertyValue("subarray_mode"))

    def clear_roi(self):
        """sets roi to full size"""
        # print('cealr CALL')
        self.roi_is = [0, 0, 2048, 2048]
        self.roi_set_roi(0, 0, 2048, 2048)

    @property
    def trigger_delay(self):
        self.delay_ms = self.hcam.getPropertyValue("trigger_delay")
        print("trigger delay is {} s".format(self.delay_ms))
        return self.delay_ms

    @trigger_delay.setter
    def trigger_delay(self, delay):
        self.trigg_delay(delay)
        print("trigger delay set to {} s".format(delay))

    def trigg_delay(self, delay):
        self.hcam.setPropertyValue("trigger_delay", delay)
        self.delay_ms = delay

    @property
    def trigger_mode(self):
        self.trigg_mode = self.hcam.getPropertyValue("trigger_source")
        return self.trigg_mode

    @trigger_mode.setter
    def trigger_mode(self, trig_md):

        self.hcam.setPropertyValue("trigger_source", trig_md)
        if trig_md == 2:
            self.hcam.setPropertyValue("trigger_polarity", 1)  # 1 is 'negative'
            self.hcam.setPropertyValue("trigger_delay", 0)  # init delay not
            self.hcam.setPropertyValue("trigger_active", 1)  # 1 is 'edge'
            self.hcam.setPropertyValue("trigger_times", 1)
        self.trigg_mode = trig_md

    @property
    def exposure(self):
        e = self.hcam.getPropertyValue("exposure_time")
        # print("exposure is {}".format(e))
        return e

    @exposure.setter
    def exposure(self, e):
        print("set exposure to {}".format(e))
        self.hcam.setPropertyValue("exposure_time", e)
        self.framerate = 1/e

    def get_grey_values(self):
        """
        :rtype: np.ndarray
        :return: 1D numpy array of grey values
        """

        [frame, dim] = self.hcam.getFrames()
        # print(frame)
        grey_values = frame[0].getData()
        return grey_values

    def get_all_frames(self):
        """
        :rtype: np.ndarray
        :return: all captured frames
        """

        [frames, dim] = self.hcam.getFrames()
        # print("frame dim is: {}".format(dim))
        # print(frames)
        return frames

    def get_grey_values_o_frames(self, all_frames, frame_no):
        """
        :rtype: np.ndarray
        :return: 1D numpy array of grey values
        """

        # print("frame no is: {}".format(frame_no))
        grey_values = all_frames[frame_no].getData()
        return grey_values

    @abc.abstractmethod
    def end(self):
        self.hcam.stopAcquisition()
        
        #super(BaseHamamatsu, self).end()

    @abc.abstractmethod
    def terminate(self) -> None:
        self.camera_open = False
        super().terminate()


class LiveHamamatsu(BaseHamamatsu): # its a thread (inherits from Camera). it runs and it emits
    def __init__(self, parent=None, **kwargs):

        super(LiveHamamatsu, self).__init__(parent, **kwargs)
        self._show_preview = True
        self.hcam.setACQMode('run_till_abort', number_frames=None)
        self.mode = "Live"
        self._acquire = False
        self.num = 1
        self.integration_mode = False
        self.img_sz_y = int(np.floor(self.cam_y / self.bin_sz))
        self.img_sz_x = int(np.floor(self.cam_x / self.bin_sz))

    def live(self):
        self.hcam.startAcquisition()
        # first_img = False
        self._show_preview = True
        while self._show_preview:
            try:
                img = np.reshape(self.get_grey_values(), (self.img_sz_y, self.img_sz_x))
                # img = np.fliplr(img)  # fixme: commented this out for roi to work proper, but in Preview .T is
                #  still used because of how setImage works, if need be chamged back jah needs reconfigure roi input
                #  as abs(2048-hpos) [I sink]
                self.im.emit(img.astype(np.float64))
            except Exception as e:
                pass

    def take_image(self):
        """the wait function should be properly implemented to get the optimum speed out of
        the multiple frame collection, the INTERNAL_FRAMEINTERVAL function should be useful
         to be implemented for this""" "DCAM_IDPROP_INTERNAL_FRAMEINTERVAL"
        ims = np.zeros((self.img_sz_y, self.img_sz_x))
        self.hcam.startAcquisition()
        time.sleep(self.exposure[0] * self.num + 0.0249 * (self.num + 2))  # frame_interval is 0.0249,
        # added 2 times more because in the limit of 1-2ms exposures frames were lost
        the_frames = self.get_all_frames()
        for i in range(self.num):
            try:
                img = np.reshape(self.get_grey_values_o_frames(the_frames, i), (self.img_sz_y, self.img_sz_x))
                ims = np.add(ims,img)
            except Exception as e:
                print(e)
        self.last_frame = ims.astype(np.float64)
        self.hcam.stopAcquisition()

    def take_average_image(self, num: int):
        """Capture and return an average image from a number of captures.

        Parameters
        ----------
        num : int
            The number of images to capture and average.

        Returns
        -------
        np.ndarray
            The averaged image as a NumPy array.
        """
        print("getting average of {} images, mpesa".format(num))
        self.num = num
        self.mode = "Acq"
        self.hcam.setACQMode('fixed_length', number_frames=self.num)
        self.take_image()
        image_aver = self.last_frame / num
        self.num = 1
        return image_aver

    def run(self):
        if self.mode == "Live":
            self._show_preview = True
            self.hcam.setACQMode('run_till_abort',number_frames=None)
            self.live()
        else:
            self._acquire = True
            self.hcam.setACQMode('fixed_length', number_frames=self.num)
            # print("cam will acquire {} frames".format(self.num))
            # self.take_image()

    def end(self):
        """ends the thread after acquisition is complete"""
        self._show_preview = False
        self.hcam.stopAcquisition()
        super(LiveHamamatsu, self).end()


if __name__ == "__main__":
    pass

# es el finAl
