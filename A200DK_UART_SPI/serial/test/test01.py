import a200dkserial


ser = a200dkserial.Serial(1, 115200)

a = ser.readline(2000)
if a:
    print(f"输入： {a}")

ser.write(b"hello\r\n")
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
ser.close()
