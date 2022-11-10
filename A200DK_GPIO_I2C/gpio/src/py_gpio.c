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
#include "gpio.h"

PyObject *high;
PyObject *low;
PyObject *in;
PyObject *out;
PyObject *gpio0;
PyObject *gpio1;
PyObject *gpio3;
PyObject *gpio4;
PyObject *gpio5;
PyObject *gpio6;
PyObject *gpio7;

void define_constants(PyObject *module)
{
   high = Py_BuildValue("i", HIGH);
   PyModule_AddObject(module, "HIGH", high);

   low = Py_BuildValue("i", LOW);
   PyModule_AddObject(module, "LOW", low);

   out = Py_BuildValue("i", OUTPUT);
   PyModule_AddObject(module, "OUTPUT", out);

   in = Py_BuildValue("i", INPUT);
   PyModule_AddObject(module, "INPUT", in);

   gpio0 = Py_BuildValue("i", GPIO0);
   PyModule_AddObject(module, "GPIO0", gpio0);
   
   gpio1 = Py_BuildValue("i", GPIO1);
   PyModule_AddObject(module, "GPIO1", gpio1);
   
   gpio3 = Py_BuildValue("i", GPIO3);
   PyModule_AddObject(module, "GPIO3", gpio3);
   
   gpio4 = Py_BuildValue("i", GPIO4);
   PyModule_AddObject(module, "GPIO4", gpio4);
   
   gpio5 = Py_BuildValue("i", GPIO5);
   PyModule_AddObject(module, "GPIO5", gpio5);
   
   gpio6 = Py_BuildValue("i", GPIO6);
   PyModule_AddObject(module, "GPIO6", gpio6);
   
   gpio7 = Py_BuildValue("i", GPIO7);
   PyModule_AddObject(module, "GPIO7", gpio7);
}

static PyObject *py_gpio_init(PyObject *self, PyObject *args) {
  int result;
  result = gpio_init();
  return Py_BuildValue("i", result);
}

static PyObject *py_gpio_close(PyObject *self, PyObject *args) {
  int result;
  result = gpio_close();
  return Py_BuildValue("i", result);
}

static PyObject *py_setup(PyObject *self, PyObject *args) {
  int result, gpio, direction;
  if (!PyArg_ParseTuple(args, "ii", &gpio, &direction)) {
    return NULL;
  }
  result = setup(gpio, direction);
  return Py_BuildValue("i", result);
}

static PyObject *py_output(PyObject *self, PyObject *args) {
  int result, gpio, value;
  if (!PyArg_ParseTuple(args, "ii", &gpio, &value)) {
    return NULL;
  }
  result = output(gpio, value);
  return Py_BuildValue("i", result);
}

static PyObject *py_input(PyObject *self, PyObject *args) {
  int result, gpio;
  if (!PyArg_ParseTuple(args, "i", &gpio)) {
    return NULL;
  }
  result = input(gpio);
  return Py_BuildValue("i", result);
}

static PyObject *py_gpio_function(PyObject *self, PyObject *args) {
  int result, gpio;
  if (!PyArg_ParseTuple(args, "i", &gpio)) {
    return NULL;
  }
  result = gpio_function(gpio);
  return Py_BuildValue("i", result);
}

static PyObject *py_cleanup(PyObject *self, PyObject *args) {
  int result, gpio;
  if (!PyArg_ParseTuple(args, "i", &gpio)) {
    result = cleanup_all();
    return Py_BuildValue("i", result);
  }
  result = cleanup(gpio);
  return Py_BuildValue("i", result);
}

static PyObject *py_setwarnings(PyObject *self, PyObject *args) {
  int result, status;
  if (!PyArg_ParseTuple(args, "i", &status)) {
    return NULL;
  }
  result = setwarnings(status);
  return Py_BuildValue("i", result);
}

static PyMethodDef gpio_methods[] = {
  {"gpio_init", py_gpio_init, METH_VARARGS, "init gpio"},
  {"gpio_close", py_gpio_close, METH_VARARGS, "close gpio"},
  {"setup", py_setup, METH_VARARGS, "set gpio direction"},
  {"output", py_output, METH_VARARGS, "set output gpio value"},
  {"input", py_input, METH_VARARGS, "get gpio value"},
  {"gpio_function", py_gpio_function, METH_VARARGS, "get gpio function, 0 for INPUT, 1 for OUTPUT"},
  {"cleanup", py_cleanup, METH_VARARGS, "set gpio function INPUT"},
  {"setwarnings", py_setwarnings, METH_VARARGS, "enable/disable warnings"},
  { NULL, NULL, 0, NULL}
};

static struct PyModuleDef gpiomodule = {
  PyModuleDef_HEAD_INIT,
  "_a200dkgpio",           /* name of module */
  "A200DK gpio module",  /* Doc string (may be NULL) */
  -1,                 /* Size of per-interpreter state or -1 */
  gpio_methods       /* Method table */
};

PyMODINIT_FUNC PyInit__a200dkgpio(void) {
  PyObject *module = PyModule_Create(&gpiomodule);
  define_constants(module);
  return module;
}