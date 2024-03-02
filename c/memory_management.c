#include "../include/memory_management.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>  // Include this to define uintptr_t

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

void* aligned_alloc(size_t size) {
    void* ptr = NULL;
    #if defined(_MSC_VER)
    ptr = _aligned_malloc(size, MEM_ALIGNMENT);
    if (!ptr) {
        return NULL;
    }
    #else
    if (posix_memalign(&ptr, MEM_ALIGNMENT, size) != 0) {
        return NULL;
    }
    #endif
    return ptr;
}

void aligned_free(void* ptr) {
    #if defined(_MSC_VER)
    _aligned_free(ptr);
    #else
    free(ptr);
    #endif
}

void* aligned_alloc_zero(size_t size) {
    void* ptr = aligned_alloc(size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

int is_aligned(void* ptr, size_t alignment) {
    return ((uintptr_t)ptr % alignment) == 0;
}

void MemoryPool_Init(MemoryPool* pool, size_t blockSize, size_t blockCount) {
    pool->blockSize = blockSize > sizeof(MemoryBlock) ? blockSize : sizeof(MemoryBlock);
    pool->freeList = NULL;
    atomic_flag_clear(&pool->lock);

    MemoryBlock* prevBlock = NULL;
    for (size_t i = 0; i < blockCount; ++i) {
        MemoryBlock* block = (MemoryBlock*)aligned_alloc_zero(pool->blockSize);
        if (!block) {
            // Handle allocation failure by cleaning up previously allocated blocks and returning
            MemoryPool_Destroy(pool);
            return;
        }
        block->next = prevBlock;
        prevBlock = block;
    }
    pool->freeList = prevBlock;
}

void MemoryPool_Grow(MemoryPool* pool, size_t additionalBlocks) {
    // Implementation from previous response
}

void* MemoryPool_Allocate(MemoryPool* pool) {
    // Implementation from previous response
}

void MemoryPool_Deallocate(MemoryPool* pool, void* ptr) {
    // Implementation from previous response
}

void MemoryPool_Destroy(MemoryPool* pool) {
    // Implementation from previous response
}

#ifdef __CUDACC__
void* cuda_alloc(size_t size) {
    void* devPtr;
    cudaMalloc(&devPtr, size);
    return devPtr;
}

void cuda_free(void* ptr) {
    cudaFree(ptr);
}

void cuda_memcpy(void* dest, const void* src, size_t n, cudaMemcpyKind kind) {
    cudaMemcpy(dest, src, n, kind);
}

void* pinned_alloc(size_t size) {
    void* hostPtr;
    cudaHostAlloc(&hostPtr, size, cudaHostAllocDefault);
    return hostPtr;
}

void pinned_free(void* ptr) {
    cudaFreeHost(ptr);
}
#endif
