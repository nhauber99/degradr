#include <Python.h>
#include <memory>

#include "ipp.h"


PyObject* demosaic_impl(PyObject*, PyObject* args)
{
  uint64_t inRawPtr;
  uint64_t outRawPtr;
  uint64_t channels;
  uint64_t height;
  uint64_t width;
  uint64_t method;
  uint64_t bayerType;

  if (!PyArg_ParseTuple(args, "KKKKKKK", &inRawPtr, &outRawPtr, &channels, &height, &width, &method, &bayerType))
    Py_RETURN_NONE;

  auto const inPtr = reinterpret_cast<float*>(inRawPtr);
  auto const outPtr = reinterpret_cast<float*>(outRawPtr);

  size_t const size = channels * width * height;
  auto inPtrU16 = std::unique_ptr<uint16_t[]>(new uint16_t[3 * width * height]);
  auto const outPtrU16 = std::unique_ptr<uint16_t[]>(new uint16_t[2 * size]);

  for (size_t i = 0; i < width * height; ++i) inPtrU16[i] = static_cast<uint16_t>(inPtr[i]);

  int ret = -1;
  if (method == 0) {
    auto const temp = std::unique_ptr<uint16_t[]>(new uint16_t[3 * (width + 6) * 30]);
    ret = ippiDemosaicAHD_16u_C1C3R(inPtrU16.get(), IppiRect{ 0, 0, static_cast<int>(width), static_cast<int>(height) }, IppiSize{ static_cast<int>(width), static_cast<int>(height) }, static_cast<int>(width * sizeof(uint16_t)),
      outPtrU16.get(), static_cast<int>(3 * width * sizeof(uint16_t)), static_cast<IppiBayerGrid>(bayerType), temp.get(), static_cast<int>(3 * width * sizeof(uint16_t)));
  }
  else if (method == 1) {
    float scale[] = { 1.f, 1.f, 1.f, 1.f };
    ret = ippiCFAToBGRA_VNG_16u_C1C4R(inPtrU16.get(), IppiRect{ 0, 0, static_cast<int>(width), static_cast<int>(height) }, IppiSize{ static_cast<int>(width), static_cast<int>(height) }, static_cast<int>(width * sizeof(uint16_t)),
      scale, outPtrU16.get(), static_cast<int>(4 * width * sizeof(uint16_t)), static_cast<IppiBayerGrid>(bayerType));
  }
  else if (method == 2) {
    ret = ippiCFAToRGB_16u_C1C3R(inPtrU16.get(), IppiRect{ 0, 0, static_cast<int>(width), static_cast<int>(height) }, IppiSize{ static_cast<int>(width), static_cast<int>(height) }, static_cast<int>(width * sizeof(uint16_t)),
      outPtrU16.get(), static_cast<int>(3 * width * sizeof(uint16_t)), static_cast<IppiBayerGrid>(bayerType), 0);
  }
  if (ret != 0) printf("Demosaic Error %d", ret);
  if (method == 1)
  {
    for (size_t r = 0; r < height; ++r)
      for (size_t c = 0; c < width; ++c)
        for (size_t i = 0; i < channels; ++i)
          outPtr[i * height * width + r * width + c] = static_cast<float>(outPtrU16[r * width * 4 + c * 4 + (2 - i)]);
  }
  else {
    for (size_t r = 0; r < height; ++r)
      for (size_t c = 0; c < width; ++c)
        for (size_t i = 0; i < channels; ++i)
          outPtr[i * height * width + r * width + c] = static_cast<float>(outPtrU16[r * width * channels + c * channels + i]);
  }

  Py_RETURN_NONE;
}

static PyMethodDef PyIPP_methods[] = {
  {"Demosaic", static_cast<PyCFunction>(demosaic_impl), METH_VARARGS, nullptr},
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
