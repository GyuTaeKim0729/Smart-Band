import bluepy.btle as btle
import binascii

CCCD_UUID = 0x2902
ch_noti_handle = None

class MyDelegate(btle.DefaultDelegate):
    def handleNotification(self, cHandle, data):
        if cHandle == ch_noti_handle:
            print(binascii.b2a_hex(data))

p=btle.Peripheral("E0:95:58:90:DD:9D",addrType=btle.ADDR_TYPE_RANDOM)
p.setDelegate(MyDelegate())