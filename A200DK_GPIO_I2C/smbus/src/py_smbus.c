/**
* Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "Python.h"
#include <string.h>
#include "smbus.h"

static PyObject *py_i2c_2_init(PyObject *self, PyObject *args) {
  int result;
  result = i2c_2_init();
  return Py_BuildValue("i", result);
}

static PyObject *py_i2c_2_close(PyObject *self, PyObject *args) {
  int result;
  result = i2c_2_close();
  return Py_BuildValue("i", result);
}

static PyObject *py_enable_pec(PyObject *self, PyObject *args) {
  int status, result;
  if (!PyArg_ParseTuple(args, "i", &status)) {
    return NULL;
  }
  result = enable_pec(status);
  return Py_BuildValue("i", result);
}

static PyObject *py_get_funcs(PyObject *self, PyObject *args) {
  int result;
  result = get_funcs();
  return Py_BuildValue("i", result);
}

static PyObject *py_read_byte(PyObject *self, PyObject *args) {
  int address, result;
  if (!PyArg_ParseTuple(args, "i", &address)) {
    return NULL;
  }
  result = read_byte(address);
  return Py_BuildValue("i", result);
}

static PyObject *py_write_byte(PyObject *self, PyObject *args) {
  int address, val, result;
  unsigned char value;
  if (!PyArg_ParseTuple(args, "ii", &address, &val)) {
    return NULL;
  }
  value = (unsigned char)val;
  result = write_byte(address, value);
  return Py_BuildValue("i", result);
}

static PyObject *py_read_byte_data(PyObject *self, PyObject *args) {
  int address, cmd, result;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "ii", &address, &cmd)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  result = read_byte_data(address, command);
  return Py_BuildValue("i", result);
}

static PyObject *py_write_byte_data(PyObject *self, PyObject *args) {
  int address, cmd, val, result;
  unsigned char command, value;
  if (!PyArg_ParseTuple(args, "iii", &address, &cmd, &val)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  value = (unsigned char)val;
  result = write_byte_data(address, command, value);
  return Py_BuildValue("i", result);
}

static PyObject *py_read_word_data(PyObject *self, PyObject *args) {
  int address, cmd, result;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "ii", &address, &cmd)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  result = read_word_data(address, command);
  return Py_BuildValue("i", result);
}

static PyObject *py_write_word_data(PyObject *self, PyObject *args) {
  int address, cmd, value, result;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "iii", &address, &cmd, &value)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  result = write_word_data(address, command, value);
  return Py_BuildValue("i", result);
}

static PyObject *py_read_block_data(PyObject *self, PyObject *args) {
  int address, cmd;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "ii", &address, &cmd)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  unsigned char *result = read_block_data(address, command);
  PyObject *result_list = PyList_New(0);
  for (int i = 0; i < I2C_SMBUS_BLOCK_MAX; i++)
    PyList_Append(result_list, Py_BuildValue("i", result[i]));
  return result_list;
}

static PyObject *py_write_block_data(PyObject *self, PyObject *args) {
  int address, cmd, result;
  PyObject *value_list;
  int length;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "iiO", &address, &cmd, &value_list)) {
    return NULL;
  }
  length = PyList_Size(value_list);
  command = (unsigned char)cmd;
  unsigned char *values = (unsigned char*)malloc(sizeof(unsigned char) * length);
  memset(values, 0, sizeof(values));
  for (int i = 0; i < length; i++)
  {
    int tmp;
    PyObject *list_item = PyList_GetItem(value_list, i);
    PyArg_Parse(list_item, "i", &tmp);
    values[i] = (unsigned char)tmp;
  }
  result = write_block_data(address, command, values);
  free(values);
  return Py_BuildValue("i", result);
}

static PyObject *py_write_quick(PyObject *self, PyObject *args) {
  int address, result;
  if (!PyArg_ParseTuple(args, "i", &address)) {
    return NULL;
  }
  result = write_quick(address);
  return Py_BuildValue("i", result);
}

static PyObject *py_process_call(PyObject *self, PyObject *args) {
  int address, cmd, result;
  unsigned char command;
  short value;
  if (!PyArg_ParseTuple(args, "iii", &address, &cmd, &value)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  result = process_call(address, command, value);
  return Py_BuildValue("i", result);
}

static PyObject *py_read_i2c_block_data(PyObject *self, PyObject *args) {
  int address, cmd, len;
  unsigned char command, length;
  if (!PyArg_ParseTuple(args, "iii", &address, &cmd, &len)) {
    return NULL;
  }
  command = (unsigned char)cmd;
  length = (unsigned char)len;
  unsigned char *result = read_i2c_block_data(address, command, length);
  PyObject *result_list = PyList_New(0);
  for (int i = 0; i < len; i++)
    PyList_Append(result_list, Py_BuildValue("i", result[i]));
  return result_list;
}

static PyObject *py_write_i2c_block_data(PyObject *self, PyObject *args) {
  int address, cmd, result;
  PyObject *value_list;
  int length;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "iiO", &address, &cmd, &value_list)) {
    return NULL;
  }
  length = PyList_Size(value_list);
  command = (unsigned char)cmd;
  unsigned char *values = (unsigned char*)malloc(sizeof(unsigned char) * length);
  memset(values, 0, sizeof(values));
  for (int i = 0; i < length; i++)
  {
    int tmp;
    PyObject *list_item = PyList_GetItem(value_list, i);
    PyArg_Parse(list_item, "i", &tmp);
    values[i] = (unsigned char)tmp;
  }
  result = write_i2c_block_data(address, command, values);
  free(values);
  return Py_BuildValue("i", result);
}

static PyObject *py_block_process_call(PyObject *self, PyObject *args) {
  int address, cmd;
  PyObject *value_list;
  int length;
  unsigned char command;
  if (!PyArg_ParseTuple(args, "iiO", &address, &cmd, &value_list)) {
    return NULL;
  }
  length = PyList_Size(value_list);
  command = (unsigned char)cmd;
  unsigned char *values = (unsigned char*)malloc(sizeof(unsigned char) * length);
  memset(values, 0, sizeof(values));
  for (int i = 0; i < length; i++)
  {
    int tmp;
    PyObject *list_item = PyList_GetItem(value_list, i);
    PyArg_Parse(list_item, "i", &tmp);
    values[i] = (unsigned char)tmp;
  }
  unsigned char *result = block_process_call(address, command, values);
  free(values);
  PyObject *result_list = PyList_New(0);
  for (int i = 0; i < length; i++)
    PyList_Append(result_list, Py_BuildValue("i", result[i]));
  return result_list;
}

static PyMethodDef smbus_methods[] = {
  {"i2c_2_init", py_i2c_2_init, METH_VARARGS, "init i2c-2"},
  {"i2c_2_close", py_i2c_2_close, METH_VARARGS, "close i2c-2"},
  {"enable_pec", py_enable_pec, METH_VARARGS, "enable/disable PEC"},
  {"get_funcs", py_get_funcs, METH_VARARGS, "get i2c capabilities"},
  {"read_byte", py_read_byte, METH_VARARGS, "read byte transaction"},
  {"write_byte", py_write_byte, METH_VARARGS, "write byte transaction"},
  {"read_byte_data", py_read_byte_data, METH_VARARGS, "read byte data transaction"},
  {"write_byte_data", py_write_byte_data, METH_VARARGS, "write byte data transaction"},
  {"read_word_data", py_read_word_data, METH_VARARGS, "read word data transaction"},
  {"write_word_data", py_write_word_data, METH_VARARGS, "write block data transaction"},
  {"read_block_data", py_read_block_data, METH_VARARGS, "read block data transaction"},
  {"write_block_data", py_write_block_data, METH_VARARGS, "write block data transaction"},
  {"write_quick", py_write_quick, METH_VARARGS, "write quick transaction"},
  {"process_call", py_process_call, METH_VARARGS, "process call"},
  {"read_i2c_block_data", py_read_i2c_block_data, METH_VARARGS, "read i2c block data transaction"},
  {"write_i2c_block_data", py_write_i2c_block_data, METH_VARARGS, "write i2c block data transaction"},
  {"block_process_call", py_block_process_call, METH_VARARGS, "block process call"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef smbusmodule = {
  PyModuleDef_HEAD_INIT,
  "_a200dksmbus",           /* name of module */
  "A200DK smbus module",  /* Doc string (may be NULL) */
  -1,                 /* Size of per-interpreter state or -1 */
  smbus_methods       /* Method table */
};

PyMODINIT_FUNC PyInit__a200dksmbus(void) {
  return PyModule_Create(&smbusmodule);
}