from bluetooth.ble import DiscoveryService

service = DiscoveryService()
dev = service.discover(2)

for address, name in device.items():
    print("name: {}, address: {}".format(name,address))