import spidev
spi = spidev.SpiDev()
spi.open(0, 0)

spi.max_speed_hz = 5000
spi.mode = 0b01
spi.writebytes2([01, 01])

spi.close()
