# _*_ coding:utf-8 _*_
# Created time : 3/16 2018
# Author: JosiahMg
# env: anaconda python3.6

import subprocess
import os

# 两种截屏操作 方法1 速度更快些
def android_screenshot(way):
    if way == 1:
        process = subprocess.Popen('adb shell screencap -p',shell=True, stdout=subprocess.PIPE)
        binary_screenshot = process.stdout.read()
        binary_screenshot = binary_screenshot.replace(b'\r\n', b'\n')
        #binary_screenshot = binary_screenshot.replace(b'\r\r\n', b'\n')
        f = open('autojump.png', 'wb')
        f.write(binary_screenshot)
        f.close()
    elif way == 2:
        os.system('adb shell screencap -p /sdcard/autojump.png')
        os.system('adb pull /sdcard/autojump.png .')