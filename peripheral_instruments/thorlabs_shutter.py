import time

import pyvisa as visa
# from PyQt5 import QtCore
from colorama import Fore, Style  # , Back


class Shutter():
    """this is the class for the thorlabs sc10 shutter controller.
       Currently, it only turns it on and off, but the sdk is available
        and implementation appears straightforward"""

    def __init__(self):
    # def __init__(self, parent=None):
        # super(Shutter, self).__init__(parent)  # requested when threaded, a ha a a
        self.rm = visa.ResourceManager()
        self.my_instrument = self.rm.open_resource('ASRL3::INSTR')
        self.my_instrument.read_termination = '\r'
        self.my_instrument.write_termination = '\r'
        self.my_instrument.query('id?')
        self.shut_state = 0
        self.action = 'check'
        self.coloz = [Fore.LIGHTRED_EX, Fore.LIGHTGREEN_EX]
        self.string_state = ["off", 'on']

        try:
            # my_instrument.write('id?')
            identity = self.my_instrument.read()
            print("Resource is" + str(self.my_instrument))
            print("instrument is: " + identity + '\n')
        except visa.VisaIOError:
            print('No connection to: ' + str(self.my_instrument))

    def shutter_state(self):
        """check shutter state 0 is off 1 is on"""
        # warning says read is not available but it does not work without it
        self.my_instrument.query('ens?')
        time.sleep(0.4)
        # print(Fore.LIGHTGREEN_EX + "shutState psotQUERY" + Style.RESET_ALL)
        self.shut_state = int(self.my_instrument.read())  # value is returned as str
        print(self.coloz[self.shut_state] + "shutter is found {}".format(self.string_state[self.shut_state])
              + Style.RESET_ALL)
        # self.end()

    def shutter_enable(self):
        """change shutter state"""
        self.my_instrument.query('ens')
        self.shutter_state()
        print(self.coloz[self.shut_state] + "shutter is {}".format(self.string_state[self.shut_state])
              + Style.RESET_ALL)
        # print(Fore.BLUE + "shutEnable end" + Style.RESET_ALL)
        # self.end()

    def run(self):
        if self.action == 'check':
            self.shutter_state()
        else:
            self.shutter_enable()

    def end(self):
        """ends the thread after acquisition is complete"""
        super(Shutter, self).quit()


# instantiate
shutter = Shutter()

# es el finAl
