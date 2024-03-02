#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Ensure alignment is defined and is adequate for SIMD and application requirements
#ifndef MEM_ALIGNMENT
#define MEM_ALIGNMENT 64 // Example alignment, adjust as necessary
#endif

void* aligned_alloc(size_t size);
void aligned_free(void* ptr);
void* aligned_alloc_zero(size_t size);
int is_aligned(void* ptr, size_t alignment); // Check if the pointer is aligned to the specified alignment

// Memory Pool structure
typedef struct MemoryBlock {
    struct MemoryBlock* next;
} MemoryBlock;

typedef struct MemoryPool {
    size_t blockSize;
    MemoryBlock* freeList;
    atomic_flag lock;
} MemoryPool;

void MemoryPool_Init(MemoryPool* pool, size_t blockSize, size_t blockCount);
void MemoryPool_Grow(MemoryPool* pool, size_t additionalBlocks);
void* MemoryPool_Allocate(MemoryPool* pool);
void MemoryPool_Deallocate(MemoryPool* pool, void* ptr);
void MemoryPool_Destroy(MemoryPool* pool);

#ifdef __CUDACC__
void* cuda_alloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind);
void* pinned_alloc(size_t size);
void pinned_free(void* ptr);
#endif

#endif // MEMORY_MANAGEMENT_H
