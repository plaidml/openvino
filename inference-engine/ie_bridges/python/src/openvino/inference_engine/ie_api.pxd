from .cimport ie_api_impl_defs as C
from .ie_api_impl_defs cimport CBlob, CTensorDesc, InputInfo, CPreProcessChannel, CPreProcessInfo

from pathlib import Path

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr

cdef class Blob:
    cdef CBlob.Ptr _ptr
    cdef public object _array_data
    cdef public object _initial_shape

cdef class BlobBuffer:
    cdef CBlob.Ptr ptr
    cdef char*format
    cdef vector[Py_ssize_t] shape
    cdef vector[Py_ssize_t] strides
    cdef reset(self, CBlob.Ptr &, vector[size_t] representation_shape = ?)
    cdef char*_get_blob_format(self, const CTensorDesc & desc)

    cdef public:
        total_stride, item_size

cdef class InferRequest:
    cdef C.InferRequestWrap *impl

    cpdef BlobBuffer _get_blob_buffer(self, const string & blob_name)

    cpdef infer(self, inputs = ?)
    cpdef async_infer(self, inputs = ?)
    cpdef wait(self, timeout = ?)
    cpdef get_perf_counts(self)
    cdef void user_callback(self, int status) with gil
    cdef public:
        _inputs_list, _outputs_list, _py_callback, _py_data, _py_callback_used, _py_callback_called, _user_blobs

cdef class IENetwork:
    cdef C.IENetwork impl

cdef class ExecutableNetwork:
    cdef unique_ptr[C.IEExecNetwork] impl
    cdef C.IEPlugin plugin_impl
    cdef C.IECore ie_core_impl
    cpdef wait(self, num_requests = ?, timeout = ?)
    cpdef get_idle_request_id(self)
    cdef public:
        _requests, _infer_requests

cdef class IEPlugin:
    cdef C.IEPlugin impl
    cpdef ExecutableNetwork load(self, IENetwork network, int num_requests = ?, config = ?)
    cpdef void set_config(self, config)
    cpdef void add_cpu_extension(self, str extension_path) except *
    cpdef void set_initial_affinity(self, IENetwork network) except *
    cpdef set get_supported_layers(self, IENetwork net)

cdef class LayersStatsMap(dict):
    cdef C.IENetwork net_impl

cdef class IECore:
    cdef C.IECore impl
    cpdef IENetwork read_network(self, model : [str, bytes, Path], weights : [str, bytes, Path] = ?, bool init_from_buffer = ?)
    cpdef ExecutableNetwork load_network(self, IENetwork network, str device_name, config = ?, int num_requests = ?)
    cpdef ExecutableNetwork import_network(self, str model_file, str device_name, config = ?, int num_requests = ?)


cdef class DataPtr:
    cdef C.DataPtr _ptr

cdef class CDataPtr:
    cdef C.CDataPtr _ptr

cdef class IENetLayer:
    cdef C.CNNLayerPtr _ptr

cdef class TensorDesc:
    cdef C.CTensorDesc impl

cdef class InputInfoPtr:
    cdef InputInfo.Ptr _ptr

cdef class InputInfoCPtr:
    cdef InputInfo.CPtr _ptr

cdef class PreProcessInfo:
    cdef CPreProcessInfo* _ptr

cdef class PreProcessChannel:
    cdef CPreProcessChannel.Ptr _ptr
