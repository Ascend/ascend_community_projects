import a200dkserial


ser = a200dkserial.Serial(1,115200)  # open serial port

ser.write(b"hello")     # write a string

while True:
    a = ser.read(9)
    if(a is not None):
        print(a)
        break;

while True:
    a = ser.readline()
    if(a is not None):
        print(a)
        break;

ser.input_waiting(3)
ser.output_waiting(3)
ser.poll(10)
print(ser.baudrate)
print(ser.databits)
print(ser.parity)
print(ser.xonxoff)
print(ser.rtscts)
print(ser.vtime)
print(ser.vmin)
print(ser.stopbits)
ser.close()             # close port
