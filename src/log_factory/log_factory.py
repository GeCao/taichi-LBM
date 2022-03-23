import os
import time
from ..utils import MessageAttribute


class LogFactory:
    def __init__(self, core_management, log_to_disk):
        self.core_management = core_management
        self.log_to_disk = log_to_disk

        self.root_path = self.core_management.root_path
        self.log_path = None

        self.file_ptr = None

        self.initialized = False

    def initialization(self):
        if self.log_to_disk:
            if not os.path.exists(os.path.join(self.root_path, "../log")):
                os.mkdir(os.path.join(self.root_path, "../log"))

            str_curr_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))

            self.file_ptr = open(os.path.join(self.root_path, "../log", str("Log-") + str_curr_time + ".txt"), 'w')
            self.file_ptr.close()

            self.log_path = os.path.join(self.root_path, "../log", str("Log-") + str_curr_time + ".txt")

            self.file_ptr = open(self.log_path, "a+")
        else:
            self.log_path = None

        self.initialized = True

    def WarnLog(self, sentences=""):
        self.Slog(message_attribute=MessageAttribute.EWarn, sentences=sentences)

    def InfoLog(self, sentences=""):
        self.Slog(message_attribute=MessageAttribute.EInfo, sentences=sentences)

    def ErrorLog(self, sentences=""):
        self.Slog(message_attribute=MessageAttribute.EError, sentences=sentences)

    def Slog(self, message_attribute=MessageAttribute(0), sentences=""):
        str_curr_time = time.strftime('[%Y-%m-%d %H:%M:%S] (', time.localtime(time.time()))
        prefix_str = ""
        final_str = ""
        if message_attribute.value == MessageAttribute.EWarn.value:
            # Set Font as yellow
            prefix_str = '\033[1;33m'
            final_str = '\033[0m'
        elif message_attribute.value == MessageAttribute.EError.value:
            # Set Font as Red
            prefix_str = '\033[1;31m'
            final_str = '\033[0m'
        elif message_attribute.value == MessageAttribute.EInfo.value:
            # Set Font as Red
            prefix_str = '\033[1;32m'
            final_str = '\033[0m'
        print(prefix_str + str_curr_time + message_attribute.name + ') ' + sentences + final_str)
        if self.log_to_disk and self.initialized:
            self.file_ptr.write(str_curr_time + message_attribute.name + ' ' + sentences + '\n')

    def kill(self):
        self.Slog(MessageAttribute.EError, sentences="Kill the Log Factory")
        if self.log_to_disk and self.initialized:
            if self.file_ptr is not None:
                self.file_ptr.close()