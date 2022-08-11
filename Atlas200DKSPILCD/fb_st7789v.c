// SPDX-License-Identifier: GPL-2.0+

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

/*
 * FB driver for the ST7789V LCD Controller
 *
 * Copyright (C) 2015 Dennis Menschel
 */

#include <linux/bitops.h>
#include <linux/delay.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <video/mipi_display.h>

#include "fbtft.h"

#define DRVNAME "fb_st7789v"

#define DEFAULT_GAMMA \
	"1F 1A 18 0A 0F 06 45 87 32 0A 07 02 07 05 00\n" \
	"00 25 27 05 10 09 3A 78 4D 05 18 0D 38 3A 1F"

#define MADCTL_BGR BIT(3) /* bitmask for RGB/BGR order */
#define MADCTL_MV BIT(5) /* bitmask for page/column order */
#define MADCTL_MX BIT(6) /* bitmask for column address order */
#define MADCTL_MY BIT(7) /* bitmask for page address order */

/**
 * init_display() - initialize the display controller
 *
 * @par: FBTFT parameter object
 *
 * Most of the commands in this init function set their parameters to the
 * same default values which are already in place after the display has been
 * powered up. (The main exception to this rule is the pixel format which
 * would default to 18 instead of 16 bit per pixel.)
 * Nonetheless, this sequence can be used as a template for concrete
 * displays which usually need some adjustments.
 *
 * Return: 0 on success, < 0 if error occurred.
 */
static int init_display(struct fbtft_par *par)
{
	write_reg(par, 0x36, 0x00);
	write_reg(par, 0x3A, 0x05);
	write_reg(par, 0xB2, 0x0C, 0x0C, 0x00, 0x33, 0x33);
	write_reg(par, 0xB7, 0x35);
	write_reg(par, 0xBB, 0x1F);
	write_reg(par, 0xC0, 0x2C);
	write_reg(par, 0xC2, 0x01);
	write_reg(par, 0xC3, 0x12);
	write_reg(par, 0xC4, 0x20);
	write_reg(par, 0xC6, 0x0F);
	write_reg(par, 0xD0, 0xA4, 0xA1);
	write_reg(par, 0xE0, 0xD0, 0x08, 0x11, 0x08, 0x0C, 0x15, 0x39, 0x33, 0x50, 0x36, 0x13, 0x14, 0x29, 0x2D);
	write_reg(par, 0xE1, 0xD0, 0x08, 0x10, 0x08, 0x06, 0x06, 0x39, 0x44, 0x51, 0x0B, 0x16, 0x14, 0x2F, 0x31);
	// write_reg(par,0x21);	// 反色
	write_reg(par, 0x11);
	write_reg(par, 0x29);
  	return 0;
}

/**
 * set_var() - apply LCD properties like rotation and BGR mode
 *
 * @par: FBTFT parameter object
 *
 * Return: 0 on success, < 0 if error occurred.
 */
static int set_var(struct fbtft_par *par)
{
	u8 madctl_par = 0;

    int angle_0 = 0, angle_90 = 90, angle_180 = 180, angle_270 = 270;
	if (par->bgr)
		madctl_par |= MADCTL_BGR;
	switch (par->info->var.rotate) {
	case angle_0:
		break;
	case angle_90:
		madctl_par |= (MADCTL_MV | MADCTL_MY);
		break;
	case angle_180:
		madctl_par |= (MADCTL_MX | MADCTL_MY);
		break;
	case angle_270:
		madctl_par |= (MADCTL_MV | MADCTL_MX);
		break;
	default:
		return -EINVAL;
	}
	write_reg(par, MIPI_DCS_SET_ADDRESS_MODE, madctl_par);
	return 0;
}

/**
 * blank() - blank the display
 *
 * @par: FBTFT parameter object
 * @on: whether to enable or disable blanking the display
 *
 * Return: 0 on success, < 0 if error occurred.
 */
static int blank(struct fbtft_par *par, bool on)
{
	if (on)
		write_reg(par, MIPI_DCS_SET_DISPLAY_OFF);
	else
		write_reg(par, MIPI_DCS_SET_DISPLAY_ON);
	return 0;
}

static struct fbtft_display display = {
	.regwidth = 8,
	.width = 240,
	.height = 320,
	.gamma_num = 2,
	.gamma_len = 15,
	.gamma = DEFAULT_GAMMA,
	.fbtftops = {
		.init_display = init_display,
		.set_var = set_var,
		.blank = blank,
	},
};

FBTFT_REGISTER_DRIVER(DRVNAME, "sitronix,st7789v", &display);

MODULE_ALIAS("spi:" DRVNAME);
MODULE_ALIAS("platform:" DRVNAME);
MODULE_ALIAS("spi:st7789v");
MODULE_ALIAS("platform:st7789v");

MODULE_DESCRIPTION("FB driver for the ST7789V LCD Controller");
MODULE_AUTHOR("ABC");
MODULE_LICENSE("GPL");
