#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../include/memory_management.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdint.h>

void* aligned_alloc(size_t size) {
    void* ptr = NULL;
    #if defined(_MSC_VER)
    ptr = _aligned_malloc(size, MEM_ALIGNMENT);
    #else
    if (posix_memalign(&ptr, MEM_ALIGNMENT, size) != 0) {
        ptr = NULL;
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

void MemoryPool_Init(MemoryPool* pool, size_t blockSize, size_t blockCount, int world_rank, int world_size) {
    // Calculate the portion of the pool this process will initialize, accounting for load balancing
    size_t blocksPerProcess = blockCount / world_size;
    size_t extraBlocks = blockCount % world_size;
    size_t startBlock = world_rank * blocksPerProcess + (world_rank < extraBlocks ? world_rank : extraBlocks);
    size_t endBlock = startBlock + blocksPerProcess + (world_rank < extraBlocks ? 1 : 0);

    pool->blockSize = blockSize > sizeof(MemoryBlock) ? blockSize : sizeof(MemoryBlock);
    pool->freeList = NULL;
    atomic_flag_clear(&pool->lock);

    int allocationFailure = 0;

    #pragma omp parallel
    {
        MemoryBlock* localFreeList = NULL;

        #pragma omp for nowait
        for (size_t i = startBlock; i < endBlock; ++i) {
            MemoryBlock* block = (MemoryBlock*)aligned_alloc_zero(pool->blockSize);
            if (!block) {
                #pragma omp atomic write
                allocationFailure = 1;  // Indicate allocation failure
                continue;
            }
            block->next = localFreeList;
            localFreeList = block;
        }

        // Thread-safe linking of blocks initialized by this thread to the global free list
        #pragma omp critical
        {
            MemoryBlock* last = localFreeList;
            while (last && last->next) {
                last = last->next;  // Find the last block in the local list
            }
            if (last) {
                last->next = pool->freeList;  // Link the local list to the global list
                pool->freeList = localFreeList;
            }
        }
    }

    if (allocationFailure) {
        MemoryPool_Destroy(pool);  // Clean up in case of allocation failure
    }
}

void MemoryPool_Grow(MemoryPool* pool, size_t additionalBlocks) {
    if (additionalBlocks == 0) return; // No need to grow if no additional blocks are requested

    while (atomic_flag_test_and_set(&pool->lock)) {
        #if !defined(_MSC_VER)
        sched_yield(); // Yield to other threads on POSIX-compliant systems
        #endif
    }

    MemoryBlock* lastBlock = pool->freeList;
    if (lastBlock) {
        // Navigate to the end of the current free list
        while (lastBlock->next) {
            lastBlock = lastBlock->next;
        }
    }

    // Add new blocks to the end of the free list
    for (size_t i = 0; i < additionalBlocks; ++i) {
        MemoryBlock* block = (MemoryBlock*)aligned_alloc_zero(pool->blockSize);
        if (!block) {
            // If allocation fails, we don't undo what's already added but stop further additions
            break;
        }

        if (lastBlock) {
            // Append to the existing free list
            lastBlock->next = block;
        } else {
            // The free list was empty, so start a new one
            pool->freeList = block;
        }

        lastBlock = block; // Move the lastBlock pointer to the new end of the list
    }

    atomic_flag_clear(&pool->lock);
}

void* MemoryPool_Allocate(MemoryPool* pool) {
    while (atomic_flag_test_and_set(&pool->lock)) {
        #if !defined(_MSC_VER)
        sched_yield(); // POSIX-compliant yield
        #endif
    }

    if (!pool->freeList) {
        atomic_flag_clear(&pool->lock);
        return NULL; // Consider growing the pool here if automatic resizing is desired
    }

    MemoryBlock* block = pool->freeList;
    pool->freeList = block->next;
    atomic_flag_clear(&pool->lock);

    return (void*)block;
}

void MemoryPool_Deallocate(MemoryPool* pool, void* ptr) {
    while (atomic_flag_test_and_set(&pool->lock)) {
        #if !defined(_MSC_VER)
        sched_yield();
        #endif
    }

    MemoryBlock* block = (MemoryBlock*)ptr;
    block->next = pool->freeList;
    pool->freeList = block;

    atomic_flag_clear(&pool->lock);
}

void MemoryPool_Destroy(MemoryPool* pool) {
    MemoryBlock* block = pool->freeList;
    while (block) {
        MemoryBlock* next = block->next;
        aligned_free(block);
        block = next;
    }
    pool->freeList = NULL;
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
