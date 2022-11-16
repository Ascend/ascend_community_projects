# 基于昇腾A200DK适配UART/SPI操作库pyserial/spidev

## 1 概述

### 1.1 概要描述

基于昇腾A200DK适配UART/SPI操作库pyserial/spidev，UART接口与树莓派的GPIO操作模块pyserial保持一致，SPI操作接口与python的spidev模块保持一致。

### 1.2 实现流程

    1、环境搭建：Atlask 200 DK
    2、用C实现pyserial/spidev接口
    3、绑定到python接口
    4、用外设或示波器测试

## 2 方案介绍

### 2.1 Atlas 200 DK UART/SPI介绍

#### 2.1.1UART

UART0是8脚和10脚，用于Ascend 310的默认调试串口（console），波特率115200。

UART1是16和18脚，可以用于扩展及与其他模块通信。

UART-Hi3559是38和40脚，用于MIPI-CSI接口接入芯片Hi3559调试，波特率115200。

表2-1-1 UART相关参数：

|   名称   | 管脚编号 |
| :------: | :------: |
| UART0_TX |    8     |
| UART0_RX |    10    |
| UART1_TX |    16    |
| UART1_RX |    18    |

#### 2.1.2SPI

SPI-CS0、SPI-CLK、SPI-MISO、SPI-MOSI四线SPI接口可以外接各种传感器，只支持master模式。

表2-1-2 SPI相关参数：

|   名称   | 管脚编号 |
| :------: | :------: |
| SPI-MOSI |    19    |
| SPI-MISO |    21    |
| SPI-CLK  |    23    |
|  SPI-CS  |    24    |

### 2.2 pyserial模块介绍

操作

```
(1)open、serial_open_advanced：设置串口参数并打开。
(2)close：关闭串口
(3)fileno：读取串口的文件描述符
(4)read、readline：读取字符串
(5)write：向串口写字符串
(6)flush：刷新缓存
(7)input_waiting、output_waiting
(8)poll:延迟轮询查询串口
```



属性

```
baudrate：波特率
bytesize：字节大小
parity：校验位
stopbits：停止位
timeout：读超时设置
writeTimeout：写超时
xonxoff：软件流控
rtscts：硬件流控
dsrdtr：硬件流控 
```



### 2.3 spidev模块介绍

操作

```
(1)spi_open、spi_open_advanced：设置SPI参数并打开。
(2)spi_xfer、spi_xfer2、spi_xfer3：传输
(3)spi_read：读n个字节
(4)spi_write：写n个字节
```

属性

```
mode
max_speed
bit_order
bits_per_word
extra_flags
cshigh
loop
no_cs
```



### 2.4 代码目录结构与说明

```
├── serial
│   ├── a200dkserial.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── setup.py
│   ├── src
│   │   ├── 200dk_serial.h
│   │   ├── c_serial.c
│   │   └── py_serial.c
│   └── test
│       ├── Makefile
│       ├── serialtest
│       ├── serial_test.c
│       └── test01.py
├── spi
│   ├── setup.py
│   ├── spidev.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── src
│   │   ├── a200dkspi.h
│   │   ├── c_spidev.c
│   │   └── py_spidev.c
│   └── test
│       ├── Makefile
│       ├── spitest
│       ├── test01.c
│       └── test01.py
├── README.md
└── 操作文档.docx
```

## 3 开发准备

### 3.1 环境依赖说明

|  软件名称  |     版本      |
| :--------: | :-----------: |
|   Ubuntu   |  18.04.1 LTS  |
|    GCC     |     7.5.0     |
|   Python   |    3.7.13     |
|    CANN    | 5.0.5alpha001 |
| 固件与驱动 |    1.0.12     |

### 3.2 环境搭建

见Atlas200dk-MindXSDK 开发环境搭建一站式导航：
https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk%E5%BC%80%E5%8F%91%E6%9D%BF%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

### 3.3 获取UART/SPI操作权限

1.见 `操作文档.docx`。/dev/下出现spidev0.0和ttyAMA1即可。

2.su root，然后输入密码，切换成root用户。

3.vim /etc/rc.local 在exit0前增加如下指令：

    chown -R HwHiAiUser /dev/spidev0.0
    chown -R HwHiAiUser /dev/ttyAMA1
    usermod -aG HwHiAiUser HwHiAiUser

4.重启运行环境。

## 4 编译与运行

### 4.1 Serial C示例步骤

**步骤1** 执行如下命令，进入到GPIO代码资源目录下。

    cd serial/src

**步骤2** 执行如下命令，将smbus.c编译成.so文件。

    gcc c_serial.c -fPIC -shared -o libserial.so

**步骤3** 执行如下命令，在动态库配置文件ld.so.conf文件中添加路径：/usr/local/lib

    vi /etc/ld.so.conf

**步骤4** 执行如下命令，将libsmbus.so复制到路径/usr/local/lib下。

    cp libserial.so /usr/local/lib

**步骤5** 执行如下命令，让动态链接库为系统所共享。

    ldconfig

**步骤6** 执行如下命令，将200dk_serial.h复制到路径/usr/include下。

    cp 200dk_serial.h /usr/include

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译测试代码。

    gcc serial_test.c -L. -lserial -o test

**步骤9** 执行如下命令，运行测试代码。

    ./test

### 4.2 Serial Python示例步骤

**步骤1** 执行如下命令，创建一个python版本为3.7的conda环境py37。

    conda create -n py37 python=3.7

**步骤2** 执行如下命令，激活环境py37。

    conda activate py37

**步骤3** 执行如下命令，进入到GPIO项目目录下。

    cd serial

**步骤4** 执行如下命令，运行setup.py，生成whl包。

    python setup.py bdist_wheel

**步骤5** 执行如下命令，进入到dist目录下。

    cd dist

**步骤6** 执行如下命令，安装whl包。

    pip install a200dkserial-0.0.1-cp37-cp37m-linux_aarch64.whl

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译并运行测试代码。

    python test01.py

### 4.3 SPI C示例步骤

**步骤1** 执行如下命令，进入到I2C代码资源目录下。

    cd spi/src

**步骤2** 执行如下命令，将smbus.c编译成.so文件。

    gcc c_spidev.c -fPIC -shared -o libspidev.so

**步骤3** 执行如下命令，在动态库配置文件ld.so.conf文件中添加路径：/usr/local/lib

    vi /etc/ld.so.conf

**步骤4** 执行如下命令，将libsmbus.so复制到路径/usr/local/lib下。

    cp libspidev.so /usr/local/lib

**步骤5** 执行如下命令，让动态链接库为系统所共享。

    ldconfig

**步骤6** 执行如下命令，将gpio.h复制到路径/usr/include下。。

    cp a200dkspi.h /usr/include

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译测试代码。

    gcc spitest.c -L. -lspidev -o test

**步骤9** 执行如下命令，运行测试代码。

    ./serial_test

### 4.4 SPI Python示例步骤

**步骤1** 执行如下命令，创建一个python版本为3.7的conda环境py37。

    conda create -n py37 python=3.7

**步骤2** 执行如下命令，激活环境py37。

    conda activate py37

**步骤3** 执行如下命令，进入到I2C项目目录下。

    cd spi

**步骤4** 执行如下命令，运行setup.py，生成whl包。

    python setup.py bdist_wheel

**步骤5** 执行如下命令，进入到dist目录下。

    cd dist

**步骤6** 执行如下命令，安装whl包。

    pip install a200dkspidev-0.0.1-cp37-cp37m-linux_aarch64.whl

**步骤7** 执行如下命令，进入测试文件。

    cd ../test

**步骤8** 执行如下命令，编译并运行测试代码。

    python test01.py

## 5 参考链接

> Atlas200dk-MindXSDK 开发环境搭建一站式导航：
> https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk%E5%BC%80%E5%8F%91%E6%9D%BF%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

> Atlas 200 DK开发者套件 1.0.13 外围设备驱动接口操作指南：
> https://www.hiascend.com/document/detail/zh/Atlas200DKDeveloperKit/1013/peripheralref/atlaspd_07_0001.html

> 树莓派的pyserial：https://pypi.org/project/pyserial/#files

> python的SPI操作模块spidev：https://pypi.org/project/spidev/#files

> Atlas200DK各硬件接口的使用样例：
> https://gitee.com/ascend/samples/tree/master/cplusplus/level1_single_api/5_200dk_peripheral