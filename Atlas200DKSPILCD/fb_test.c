"""
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/fb.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
 
void set_pixel_color(int x, int y, unsigned int color) // 将坐标为(x,y)的像素设为指定颜色
{
	int index = y * line_length + x * bytes_per_pixel;
 
	*(unsigned int *)(fb_base + index) = color;
}
 
int main(void)
{
    struct fb_var_screeninfo vinfo = {0}; // 定义结构体变量，读取屏幕信息时用来记录屏幕可变信息的
    unsigned char *fb_base; // 显存的基地址
    unsigned int line_length;
    unsigned int bytes_per_pixel; // 每像素占用字节数
	int i;
	int xs = 70, ys = 110, xe = 170, ye = 210;
	int bits_per_byte = 8;
	int fd_fb;
	int frame_buffer_size; // 显存大小
	fd_fb = open("/dev/fb0", O_RDWR); // 打开fb0
	if (fd_fb < 0)
	{
		perror("open");
		return -1;
	}

	if (ioctl(fd_fb, FBIOGET_VSCREENINFO, &vinfo)) // 获取可变的参数fb_var_screeninfo的值
	{
		perror("ioctl");
		return -1;
	}

	bytes_per_pixel = vinfo.bits_per_pixel / bits_per_byte;
	line_length = vinfo.xres_virtual * bytes_per_pixel;
	frame_buffer_size = vinfo.xres_virtual * vinfo.yres_virtual * bytes_per_pixel;

	fb_base = (unsigned char *)mmap(NULL, frame_buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_fb, 0); // 通过mmap获取显存地址
	if (fb_base == (unsigned char *)-1)
	{
		printf("can't mmap\n");
		return -1;
	}

	memset(fb_base, 0xff, frame_buffer_size); // 将全屏设为白色
 
	for (i = xs; i <= xe; i++)
	{
		set_pixel_color(i, ys, 0x435c); // 皇家蓝
		set_pixel_color(i, ye, 0xf800); // 红
	}
	for (i = ys; i <= ye; i++)
	{
		set_pixel_color(xs, i, 0x3d8e); // 绿
		set_pixel_color(xe, i, 0xfea0); // 金
	}

	munmap(fb_base, frame_buffer_size); // 释放显存映射
	close(fd_fb);
 
	return 0;
}
