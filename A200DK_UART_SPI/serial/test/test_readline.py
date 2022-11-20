import a200dkserial


ser = a200dkserial.Serial(1, 115200)

a = ser.readline(2000)
if a:
    print(f"输入： {a}")

ser.write(b"hello\r\n")
ser.input_waiting(3)
ser.output_waiting(3)
ser.poll(10)
print(f"波特率： {ser.baudrate}")
print(f"数据位： {ser.databits}")
print(f"奇偶校验： {ser.parity}")
print(f"xonxoff： {ser.xonxoff}")
print(f"rtscts: {ser.rtscts}")
print(f"vtime: {ser.vtime}")
print(f"vim: {ser.vmin}")
print(f"停止位: {ser.stopbits}")
ser.close()
