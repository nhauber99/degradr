#include <Python.h>
#include <Windows.h>
#include <cmath>
#include <intrin.h>
#include <memory>

#include "ipp.h"


PyObject* demosaic_ahd_impl(PyObject*, PyObject* args)
{
  uint64_t inRawPtr;
  uint64_t outRawPtr;
  uint64_t channels;
  uint64_t height;
  uint64_t width;

  if (!PyArg_ParseTuple(args, "KKKKK", &inRawPtr, &outRawPtr, &channels, &height, &width))
    Py_RETURN_NONE;

  auto inPtr = reinterpret_cast<float*>(inRawPtr);
  auto outPtr = reinterpret_cast<float*>(outRawPtr);

  size_t const size = channels * width * height;
  auto inPtrU16 = std::unique_ptr<uint16_t[]>(new uint16_t[size]);
  auto outPtrU16 = std::unique_ptr<uint16_t[]>(new uint16_t[size]);
  auto temp = std::unique_ptr<uint16_t[]>(new uint16_t[(width + 6) * 30]);

  for (size_t i = 0; i < size; ++i) inPtrU16[i] = static_cast<uint16_t>(inPtr[i]);
  auto ret = ippiDemosaicAHD_16u_C1C3R(inPtrU16.get(), IppiRect{0, 0, static_cast<int>(width), static_cast<int>(height)}, IppiSize(static_cast<int>(width), static_cast<int>(height)), static_cast<int>(width) * sizeof(uint16_t), outPtrU16.get(), static_cast<int>(width) * sizeof(uint16_t),
                            ippiBayerRGGB, temp.get(), static_cast<int>(width) * sizeof(uint16_t));
  for (size_t i = 0; i < size; ++i) outPtr[i] = inPtrU16[i];

  Py_RETURN_NONE;
}

PyObject* tanh_impl(PyObject*, PyObject* o)
{
  return PyFloat_FromDouble(tanh(PyFloat_AsDouble(o)));
}

static PyMethodDef PyIPP_methods[] = {
  {"fast_tanh", static_cast<PyCFunction>(tanh_impl), METH_O, nullptr},
  {"DemosaicAHD", static_cast<PyCFunction>(demosaic_ahd_impl), METH_VARARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef PyIPP_module = {
  PyModuleDef_HEAD_INIT,
  "PyIPP", // Module name to use with Python import statements
  "Wrapps some methods of the Intel Performance Primitives", // Module description
  0,
  PyIPP_methods // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_PyIPP()
{
  return PyModule_Create(&PyIPP_module);
}
