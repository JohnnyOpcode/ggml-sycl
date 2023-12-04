#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

// so much work todo ..

#define GGML_SYCL_MAX_DEVICES       256

void   ggml_sycl_init(void);

void   ggml_sycl_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_sycl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_sycl_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_sycl_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

void * ggml_sycl_host_malloc(size_t size);
void   ggml_sycl_host_free(void * ptr);

void   ggml_sycl_free_data(const struct ggml_tensor* tensor);

void   ggml_sycl_transform_tensor(void * data, struct ggml_tensor * tensor);

void * ggml_sycl_host_malloc(size_t size);
void   ggml_sycl_host_free(void * ptr);

bool   ggml_sycl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_sycl_set_tensor_split(const float * tensor_split);
void   ggml_sycl_transform_tensor(void * data, struct ggml_tensor * tensor);
void   ggml_sycl_free_data(struct ggml_tensor * tensor);

void   ggml_sycl_assign_buffers(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers_no_scratch(struct ggml_tensor * tensor);
void   ggml_sycl_assign_buffers_force_inplace(struct ggml_tensor * tensor);

void   ggml_sycl_assign_buffers_no_alloc(struct ggml_tensor * tensor);
void   ggml_sycl_assign_scratch_offset(struct ggml_tensor * tensor, size_t offset);
void   ggml_sycl_copy_to_device(struct ggml_tensor * tensor);

void   ggml_sycl_set_main_device(int main_device);
void   ggml_sycl_set_mul_mat_q(bool mul_mat_q);
void   ggml_sycl_set_scratch_size(size_t scratch_size);
void   ggml_sycl_free_scratch(void);
bool   ggml_sycl_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);

int    ggml_sycl_get_device_count(void);
void   ggml_sycl_get_device_description(int device, char * description, size_t description_size);

ggml_backend_t ggml_backend_sycl_init(int device);

bool   ggml_backend_is_sycl(ggml_backend_t backend);
int    ggml_backend_sycl_get_device(ggml_backend_t backend);

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);

#ifdef  __cplusplus
}
#endif