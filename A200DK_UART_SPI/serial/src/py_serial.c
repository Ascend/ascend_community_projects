#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>

#include <sys/select.h>
#include <sys/ioctl.h>
#include <termios.h>

#include "structmember.h"
#include "200dk_serial.h"

#define _VERSION_ "0.0.1"
#if PY_MAJOR_VERSION < 3
#define PyLong_AS_LONG(val) PyInt_AS_LONG(val)
#define PyLong_AsLong(val) PyInt_AsLong(val)
#endif

//	Macros needed for Python 3
#ifndef PyInt_Check
#define PyInt_Check			PyLong_Check
#define PyInt_FromLong	PyLong_FromLong
#define PyInt_AsLong		PyLong_AsLong
#define PyInt_Type			PyLong_Type
#endif

#define SERIAL_MAXPATH 4096

PyDoc_STRVAR(Serial_module_doc,
	"This module defines an object type that allows uart transactions\n"
	"Because the uart device interface is opened R/W, \n"
	"Users of this module usually must have root permissions.\n");

typedef struct python_serial {
    PyObject_HEAD

    int fd;
    bool use_termios_timeout;
    uint32_t baudrate;
    unsigned int databits;
    uint32_t parity;
    unsigned int stopbits;
    bool xonxoff;
    bool rtscts;
    unsigned int vmin;
    float vtime;
}SerialObject;

static PyObject *
Serial_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	SerialObject *self;
	if ((self = (SerialObject *)type->tp_alloc(type, 0)) == NULL)
		return NULL;

    self->fd = -1;
    self->use_termios_timeout = false;
    self->baudrate = 115200;
    self->databits = 8;
    self->parity = 0;
    self->stopbits = 1;
    self->xonxoff = false;
    self->rtscts = false;
    self->vmin = 0;
    self->vtime = 0;

	Py_INCREF(self);
	return (PyObject *)self;
}

static PyObject *
Serial_close(SerialObject *self)
{
	if ((self->fd != -1) && (close(self->fd) == -1)) {
		PyErr_SetFromErrno(PyExc_IOError);
		return NULL;
	}

    self->fd = -1;
    self->use_termios_timeout = false;
    self->baudrate = 115200;
    self->databits = 8;
    self->parity = 0;
    self->stopbits = 1;
    self->xonxoff = false;
    self->rtscts = false;
    self->vmin = 0;
    self->vtime = 0;

	Py_INCREF(Py_None);
	return Py_None;
}

static void
Serial_dealloc(SerialObject *self)
{
	PyObject *ref = Serial_close(self);
	Py_XDECREF(ref);
	Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *
Serial_write(SerialObject *self, PyObject *seq)
{
	Py_ssize_t	len;
	PyObject *uni;
	char *p;

	if (!PyArg_ParseTuple(seq, "S:buf", &uni)) {
		PyErr_SetString(PyExc_TypeError,
			"failed to parse S");
		return NULL;
	}

	if(!PyBytes_Check(uni)) {
		PyErr_SetString(PyExc_TypeError,
			"failed to pass PyBytes_Check");
		return NULL;
	}

	p = PyBytes_AsString(uni);
	len = sizeof(p);

	if(serial_write(self->fd,(uint8_t *)p,(size_t)len)<0) {
		PyErr_SetString(PyExc_TypeError,
			"failed to serial_write");
		Py_DECREF(uni);
	    return NULL;
    }

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
Serial_read(SerialObject *self, PyObject *args)
{
	uint8_t	rxbuf[SERIAL_MAXPATH];
	int		status, len, timeout = 0;
	PyObject	*py_str;

	if (!PyArg_ParseTuple(args, "i|I:buf timeout", &len, &timeout))
	{
		PyErr_SetString(PyExc_TypeError,
			"failed to parse i");
		return NULL;
	}

	/*	read at least 1 byte, no more than SERIAL_MAXPATH	*/
	if (len < 1)
		len = 1;
	else if ((unsigned)len > sizeof(rxbuf))
		len = sizeof(rxbuf);

	memset(rxbuf, 0, sizeof(rxbuf));

	status = serial_read(self->fd, rxbuf, len, timeout);
	if (status < 0) {
		PyErr_SetString(PyExc_IOError,
			"failed to read");
		return NULL;
	}

	if (status <= 0) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	py_str = PyUnicode_FromStringAndSize((const char*)(char*)rxbuf,status);

	return py_str;
}

static PyObject *
Serial_readline(SerialObject *self, PyObject *args)
{
	uint8_t	rxbuf[SERIAL_MAXPATH];
	int		status, timeout_ms;
	PyObject	*py_str;

	if (!PyArg_ParseTuple(args, "|I:timeout", &timeout_ms))
	{
		PyErr_SetString(PyExc_TypeError,
			"failed to parse I (readline)");
		return NULL;
	}

    status = serial_readline(self->fd, rxbuf, SERIAL_MAXPATH,timeout_ms);

	if(status < 0) {
		PyErr_SetString(PyExc_IOError,
			"failed to read");
		return NULL;
	}

	if(status == 0) {
		Py_INCREF(Py_None);
		return Py_None;
	}

	py_str = PyUnicode_FromStringAndSize((const char*)(char*)rxbuf,status);

	return py_str;
}

static PyObject *
Serial_flush(SerialObject *self) {

    PyObject *result;

	if(serial_flush(self->fd)<0)
        result = Py_True;
    else
        result = Py_False;

	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_input_waiting(SerialObject *self, PyObject *args) {
	uint32_t count = 0;

	if (!PyArg_ParseTuple(args, "I:inputwaiting", &count))
		return NULL;
	
    PyObject *result;

	if(serial_input_waiting(self->fd, &count)<0)
        result = Py_True;
    else
        result = Py_False;

	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_output_waiting(SerialObject *self, PyObject *args) {
	uint32_t count = 0;

	if (!PyArg_ParseTuple(args, "I:outputwaiting", &count))
		return NULL;
	
    PyObject *result;

	if(serial_output_waiting(self->fd, &count)<0)
        result = Py_True;
    else
        result = Py_False;

	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_poll(SerialObject *self, PyObject *args) {
	int timeout_ms = 0;

	if (!PyArg_ParseTuple(args, "i:timeout_ms", &timeout_ms))
		return NULL;
	
    PyObject *result;

	if(serial_poll(self->fd, timeout_ms)<0)
        result = Py_True;
    else
        result = Py_False;

	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_fileno(SerialObject *self)
{
	PyObject *result = Py_BuildValue("i", self->fd);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_baudrate(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("I", self->baudrate);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_databits(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("I", self->databits);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_parity(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("I", self->parity);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_stopbits(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("I", self->stopbits);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_xonxoff(SerialObject *self, void *closure) {
	PyObject *result;
    if (self->xonxoff == true)
        result = Py_True;
    else
        result = Py_False;
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_rtscts(SerialObject *self, void *closure) {
	PyObject *result;

    if (self->rtscts == true)
        result = Py_True;
    else
        result = Py_False;

	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_vtime(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("f", self->vtime);
	Py_INCREF(result);
	return result;
}

static PyObject *
Serial_get_vmin(SerialObject *self, void *closure) {
	PyObject *result = Py_BuildValue("I", self->vmin);
	Py_INCREF(result);
	return result;
}


static int
Serial_set_baudrate(SerialObject *self, PyObject *val, void *closure)
{
	uint32_t baudrate;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
#if PY_MAJOR_VERSION < 3
	if (PyInt_Check(val)) {
		baudrate = PyInt_AS_LONG(val);
	} else
#endif
	{
		if (PyLong_Check(val)) {
			baudrate = PyLong_AsUnsignedLong(val);
		} else {
			PyErr_SetString(PyExc_TypeError,
				"string");
			return -1;
		}
	}

	self->baudrate = baudrate;
	ret = serial_set_baudrate(self->fd,baudrate);

	return ret;
}

static int
Serial_set_databits(SerialObject *self, PyObject *val, void *closure)
{
	uint32_t databits;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
#if PY_MAJOR_VERSION < 3
	if (PyInt_Check(val)) {
		databits = PyInt_AS_LONG(val);
	} else
#endif
	{
		if (PyLong_Check(val)) {
			databits = PyLong_AsUnsignedLong(val);
		} else {
			PyErr_SetString(PyExc_TypeError,
				"string");
			return -1;
		}
	}
	self->databits = databits;
	ret = serial_set_databits(self->fd,databits);
	return ret;
}

static int
Serial_set_parity(SerialObject *self, PyObject *val, void *closure)
{
	uint32_t parity;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
#if PY_MAJOR_VERSION < 3
	if (PyInt_Check(val)) {
		parity = PyInt_AS_LONG(val);
	} else
#endif
	{
		if (PyLong_Check(val)) {
			parity = PyLong_AsUnsignedLong(val);
		} else {
			PyErr_SetString(PyExc_TypeError,
				"string");
			return -1;
		}
	}
	self->parity = parity;
	ret = serial_set_parity(self->fd,parity);
	return ret;
}

static int
Serial_set_stopbits(SerialObject *self, PyObject *val, void *closure)
{
	uint32_t stopbits;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
#if PY_MAJOR_VERSION < 3
	if (PyInt_Check(val)) {
		stopbits = PyInt_AS_LONG(val);
	} else
#endif
	{
		if (PyLong_Check(val)) {
			stopbits = PyLong_AsUnsignedLong(val);
		} else {
			PyErr_SetString(PyExc_TypeError,
				"string");
			return -1;
		}
	}
	self->stopbits = stopbits;
	ret = serial_set_stopbits(self->fd,stopbits);

	return ret;
}

static int
Serial_set_xonxoff(SerialObject *self, PyObject *val, void *closure)
{
	int ret;

    if (val == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "Cannot delete attribute");
        return -1;
    }
    else if (!PyBool_Check(val)) {
        PyErr_SetString(PyExc_TypeError,
            "The xonxoff attribute must be boolean");
        return -1;
    }

	self->xonxoff = (val == Py_True) ? true : false;

	ret = serial_set_xonxoff(self->fd,self->xonxoff);
	
	return ret;
}

static int
Serial_set_rtscts(SerialObject *self, PyObject *val, void *closure)
{
	int ret;
	
    if (val == NULL) {
        PyErr_SetString(PyExc_TypeError,
            "Cannot delete attribute");
        return -1;
    }
    else if (!PyBool_Check(val)) {
        PyErr_SetString(PyExc_TypeError,
            "The xonxoff attribute must be boolean");
        return -1;
    }

	self->rtscts = (val == Py_True) ? true : false;

	ret = serial_set_rtscts(self->fd,self->rtscts);
	
	return ret;
}

static int
Serial_set_vmin(SerialObject *self, PyObject *val, void *closure)
{
	uint32_t vmin;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
#if PY_MAJOR_VERSION < 3
	if (PyInt_Check(val)) {
		vmin = PyInt_AS_LONG(val);
	} else
#endif
	{
		if (PyLong_Check(val)) {
			vmin = PyLong_AsUnsignedLong(val);
		} else {
			PyErr_SetString(PyExc_TypeError,
				"string");
			return -1;
		}
	}

	self->vmin = vmin;
	ret = serial_set_stopbits(self->fd,vmin);

	return ret;
}

static float
Serial_set_vtime(SerialObject *self, PyObject *val, void *closure)
{
	float vtime;
	int ret;

	if (val == NULL) {
		PyErr_SetString(PyExc_TypeError,
			"Cannot delete attribute");
		return -1;
	}
	if (PyFloat_Check(val)) {
		vtime = (float)PyFloat_AsDouble(val);
	} else {
		PyErr_SetString(PyExc_TypeError,
			"The vtime attribute must be an integer");
		return -1;
	}


	self->vtime = vtime;
	ret = serial_set_stopbits(self->fd,vtime);

	return ret;
}

static PyGetSetDef Serial_getset[] = {
	{"baudrate", (getter)Serial_get_baudrate, (setter)Serial_set_baudrate,
			"Serial baudrate \n"},
	{"databits", (getter)Serial_get_databits, (setter)Serial_set_databits,
			"databits\n"},
	{"parity", (getter)Serial_get_parity, (setter)Serial_set_parity,
			"parity\n"},
	{"xonxoff", (getter)Serial_get_xonxoff, (setter)Serial_set_xonxoff,
			"xonxoff\n"},
	{"rtscts", (getter)Serial_get_rtscts, (setter)Serial_set_rtscts,
			"rtscts\n"},
	{"vtime", (getter)Serial_get_vtime, (setter)Serial_set_vtime,
			"vtime\n"},
	{"vmin", (getter)Serial_get_vmin, (setter)Serial_set_vmin,
			"vmin\n"},
	{"stopbits", (getter)Serial_get_stopbits, (setter)Serial_set_stopbits,
			"stopbits\n"},
	{NULL},
};

static PyObject *
Serial_open2(SerialObject *self, PyObject *args, PyObject *kwds)
{
	int num;
	char path[SERIAL_MAXPATH];
    uint32_t baudrate,databits,parity,stopbits,vmin;
    bool xonxoff,rtscts;
    float vtime;

	static char *kwlist[] = {"num","xonxoff","rtscts","vmin","stopbits","parity","baudrate","databits","vtime",NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ippIIIIIf:open2", 
			kwlist,&num,&xonxoff,&rtscts,&baudrate,&databits,&parity,&stopbits,&vmin,&vtime))
		return NULL;
	if (snprintf(path, SERIAL_MAXPATH, "/dev/ttyAMA%d", num) >= SERIAL_MAXPATH) {
		PyErr_SetString(PyExc_OverflowError,
			"num number is invalid.");
		return NULL;
	}
	if((self->fd = serial_open_advanced(path, baudrate, databits, 
		parity, stopbits, xonxoff, rtscts))<0) {
		PyErr_SetFromErrno(PyExc_IOError);
		return NULL;
	}

    self->use_termios_timeout = false;
    self->baudrate = baudrate;
    self->databits = databits;
    self->parity = parity;
    self->stopbits = stopbits;
    self->xonxoff = xonxoff;
    self->rtscts = rtscts;
    self->vmin = 0;
    self->vtime = 0;

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject *
Serial_open(SerialObject *self, PyObject *args, PyObject *kwds)
{
	int num;
	char path[SERIAL_MAXPATH];
    uint32_t baudrate;
    //	float vtime;
	//	uint8_t tmp8;
	//	uint32_t tmp32,temp_xonxoff,temp_rtscts;
	static char *kwlist[]={"num","baudrate",NULL};
	//	Py_UNICODE *temp;
	//	char *_path;
	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iI:open", 
			kwlist,&num,&baudrate))
		return NULL;
	//	if (!PyArg_ParseTupleAndKeywords(args, kwds, "u:open", 
	//			kwlist,temp))
	//		return NULL;
	//	if(PyUnicode_Check(temp))
	//		_path = PyUnicode_AS_DATA(temp)；
		
	if (snprintf(path, SERIAL_MAXPATH, "/dev/ttyAMA%d", num) >= SERIAL_MAXPATH) {
		PyErr_SetString(PyExc_OverflowError,
			"num number is invalid.");
		return NULL;
	}

	if((self->fd = serial_open_advanced(path, baudrate, 8, 
		PARITY_NONE, 1, false, false))<0) {
		PyErr_SetFromErrno(PyExc_IOError);
		return NULL;
	}

    self->use_termios_timeout = false;
    self->baudrate = baudrate;
    self->databits = 8;
    self->parity = 0;
    self->stopbits = 1;
    self->xonxoff = false;
    self->rtscts = false;
    self->vmin = 0;
    self->vtime = 0;

	Py_INCREF(Py_None);
	return Py_None;
}

static int
Serial_init(SerialObject *self, PyObject *args, PyObject *kwds)
{
	int num = -1;
    uint32_t baudrate = -1;
	static char *kwlist[]={"num", "baudrate", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iI:__init__",
			kwlist, &num, &baudrate))
		return -1;

	if (baudrate > 0) {
		Serial_open(self, args, kwds);
		if (PyErr_Occurred())
			return -1;
	}

	return 0;
}

PyDoc_STRVAR(SerialObjectType_doc,
	"Serial([num],[baudrate]) -> Serial\n\n"
	"Return a new Serial object that is (optionally) connected to the\n"
	"specified Serial device interface.\n");

static
PyObject *Serial_enter(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

    Py_INCREF(self);
    return self;
}

static
PyObject *Serial_exit(SerialObject *self, PyObject *args)
{
    PyObject *exc_type = 0;
    PyObject *exc_value = 0;
    PyObject *traceback = 0;
    if (!PyArg_UnpackTuple(args, "__exit__", 3, 3, &exc_type, &exc_value,
                           &traceback)) {
        return 0;
    }

    Serial_close(self);
    Py_RETURN_FALSE;
}

static char to_str_buf[SERIAL_MAXPATH];

static 
PyObject *Serial_str(PyObject *self)
{
	memset(to_str_buf, 0, SERIAL_MAXPATH);
	serial_tostring(((SerialObject *)self)->fd,to_str_buf,SERIAL_MAXPATH);
    return Py_BuildValue("s", "to_str_buf");
}

static PyMethodDef Serial_methods[] = {
	{"open", (PyCFunction)Serial_open, METH_VARARGS | METH_KEYWORDS,
		"Serial_open_doc"},
	{"open2", (PyCFunction)Serial_open2, METH_VARARGS | METH_KEYWORDS,
		"Serial_open_doc"},
	{"close", (PyCFunction)Serial_close, METH_NOARGS,
		"Serial_close"},
	{"fileno", (PyCFunction)Serial_fileno, METH_NOARGS,
		"Serial_fileno"},
	{"read", (PyCFunction)Serial_read, METH_VARARGS,
		"Serial_read"},
	{"readline", (PyCFunction)Serial_readline, METH_VARARGS,
		"Serial_readline"},
	{"write", (PyCFunction)Serial_write, METH_VARARGS,
		"Serial_write"},
	{"flush", (PyCFunction)Serial_flush, METH_VARARGS,
		"Serial_flush"},
	{"input_waiting", (PyCFunction)Serial_input_waiting, METH_VARARGS,
		"Serial_input_waiting"},
	{"output_waiting", (PyCFunction)Serial_output_waiting, METH_VARARGS,
		"Serial_output_waiting"},
	{"poll", (PyCFunction)Serial_poll, METH_VARARGS,
		"Serial_poll"},
	{"__enter__", (PyCFunction)Serial_enter, METH_VARARGS,
		NULL},
	{"__exit__", (PyCFunction)Serial_exit, METH_VARARGS,
		NULL},
	{NULL},
};

static PyTypeObject SerialObjectType = {
#if PY_MAJOR_VERSION >= 3
	PyVarObject_HEAD_INIT(NULL, 0)
#else
	PyObject_HEAD_INIT(NULL)
	0,				/*	ob_size	*/
#endif
	"Serial",			/*	tp_name	*/
	sizeof(SerialObject),		/*	tp_basicsize	*/
	0,				/*	tp_itemsize	*/
	(destructor)Serial_dealloc,	/*	tp_dealloc	*/
	0,				/*	tp_print	*/
	0,				/*	tp_getattr	*/
	0,				/*	tp_setattr	*/
	0,				/*	tp_compare	*/
	0,				/*	tp_repr	*/
	0,				/*	tp_as_number	*/
	0,				/*	tp_as_sequence	*/
	0,				/*	tp_as_mapping	*/
	0,				/*	tp_hash	*/
	0,				/*	tp_call	*/
	(reprfunc)Serial_str,				/*	tp_str	*/
	0,				/*	tp_getattro	*/
	0,				/*	tp_setattro	*/
	0,				/*	tp_as_buffer	*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*	tp_flags	*/
	SerialObjectType_doc,		/*	tp_doc	*/
	0,				/*	tp_traverse	*/
	0,				/*	tp_clear	*/
	0,				/*	tp_richcompare	*/
	0,				/*	tp_weaklistoffset	*/
	0,				/*	tp_iter	*/
	0,				/*	tp_iternext	*/
	Serial_methods,			/*	tp_methods	*/
	0,				/*	tp_members	*/
	Serial_getset,			/*	tp_getset	*/
	0,				/*	tp_base	*/
	0,				/*	tp_dict	*/
	0,				/*	tp_descr_get	*/
	0,				/*	tp_descr_set	*/
	0,				/*	tp_dictoffset	*/
	(initproc)Serial_init,		/*	tp_init	*/
	0,				/*	tp_alloc	*/
	Serial_new,			/*	tp_new	*/
};

static PyMethodDef Serial_module_methods[] = {
	{NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
	PyModuleDef_HEAD_INIT,
	"a200dkserial",
	Serial_module_doc,
	-1,
	Serial_module_methods,
	NULL,
	NULL,
	NULL,
	NULL,
};
#else
#ifndef PyMODINIT_FUNC	/*	declarations for DLL import/export	*/
#define PyMODINIT_FUNC void
#endif
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_a200dkserial(void)
#else
void inita200dkserial(void)
#endif
{
	PyObject* m;

	if (PyType_Ready(&SerialObjectType) < 0)
#if PY_MAJOR_VERSION >= 3
		return NULL;
#else
		return;
#endif

#if PY_MAJOR_VERSION >= 3
	m = PyModule_Create(&moduledef);
	PyObject *version = PyUnicode_FromString(_VERSION_);
#else
	m = Py_InitModule3("a200dkserial", Serial_module_methods, Serial_module_doc);
	PyObject *version = PyString_FromString(_VERSION_);
#endif

	PyObject *dict = PyModule_GetDict(m);
	PyDict_SetItemString(dict, "__version__", version);
	Py_DECREF(version);

	Py_INCREF(&SerialObjectType);
	PyModule_AddObject(m, "Serial", (PyObject *)&SerialObjectType);

#if PY_MAJOR_VERSION >= 3
	return m;
#endif
}
