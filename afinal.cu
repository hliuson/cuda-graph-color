#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/count.h>
#include <curand_kernel.h>

static void load_edges_to_csc(const char *path,
                              std::vector<int> &colPtr,
                              std::vector<int> &rowIdx,
                              int &vertices,
                              int &edges,
                              bool zeroBaseInput = true,
                              bool csv = false
                            ) {
    std::ifstream fin(path);
    if (!fin) {
        fprintf(stderr, "Failed to open edges file: %s\n", path);
        vertices = 0;
        edges = 0;
        return;
    }

    std::vector<std::pair<int,int>> E;
    E.reserve(1<<20);
    int maxV = -1;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        
        int u, v;
        if (csv) {
            // Parse CSV format: v1,v2
            size_t commaPos = line.find(',');
            if (commaPos == std::string::npos) continue;
            try {
                u = std::stoi(line.substr(0, commaPos));
                v = std::stoi(line.substr(commaPos + 1));
            } catch (...) {
                continue;
            }
        } else {
            // Parse whitespace-separated format
            std::istringstream iss(line);
            if (!(iss >> u >> v)) continue;
        }
        
        if (!zeroBaseInput) { u -= 1; v -= 1; }
        if (u < 0 || v < 0) continue;
        // Add edge in both directions for undirected graph
        E.emplace_back(u, v);
        if (u != v) {  // Avoid duplicate self-loops
            E.emplace_back(v, u);
        }
        maxV = std::max(maxV, std::max(u, v));
    }
    edges = (int)E.size();
    vertices = (maxV >= 0) ? (maxV + 1) : 0;
    colPtr.assign(vertices + 1, 0);
    rowIdx.resize(edges);

    // Count in-degree per vertex (column)
    for (const auto &e : E) {
        int v = e.second;
        if (v >= 0 && v < vertices) colPtr[v + 1]++;
    }
    // Prefix sum to get column pointers
    for (int i = 0; i < vertices; ++i) colPtr[i + 1] += colPtr[i];

    // Temporary write positions per column
    std::vector<int> writePos = colPtr;
    for (const auto &e : E) {
        int u = e.first;
        int v = e.second;
        int pos = writePos[v]++;
        rowIdx[pos] = u;
    }

    // Sort and deduplicate row indices within each column
    int totalDeduped = 0;
    for (int c = 0; c < vertices; ++c) {
        int start = colPtr[c];
        int end = colPtr[c + 1];
        std::sort(rowIdx.begin() + start, rowIdx.begin() + end);
        auto it = std::unique(rowIdx.begin() + start, rowIdx.begin() + end);
        int newLen = (int)(it - (rowIdx.begin() + start));
        totalDeduped += (end - start) - newLen;
    }

    // Rebuild compactly if any duplicates were found
    if (totalDeduped > 0) {
        std::vector<int> newColPtr(vertices + 1, 0);
        std::vector<int> newRowIdx;
        newRowIdx.reserve(edges - totalDeduped);
        
        for (int c = 0; c < vertices; ++c) {
            int start = colPtr[c];
            int end = colPtr[c + 1];
            // Re-sort and unique since we need to do this properly
            std::sort(rowIdx.begin() + start, rowIdx.begin() + end);
            auto it = std::unique(rowIdx.begin() + start, rowIdx.begin() + end);
            int uniqueEnd = (int)(it - rowIdx.begin());
            
            newColPtr[c + 1] = newColPtr[c] + (uniqueEnd - start);
            for (int i = start; i < uniqueEnd; ++i) {
                newRowIdx.push_back(rowIdx[i]);
            }
        }
        
        colPtr.swap(newColPtr);
        rowIdx.swap(newRowIdx);
        edges = (int)rowIdx.size();
    }
}



__global__ void luby_mis_coloring_iter(int *vertex_pointers, int *adjacencies, int vertices, 
                                        int *coloring, int *candidates, int *n_colored, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    const int MAX_COLOR_WORDS = 8;  // Supports up to 256 colors
    
    for (int i = tid; i < vertices; i += gridsize) {
        candidates[i] = -1;
        if (coloring[i] != -1) continue;
        
        int is_candidate = 1;
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            if (neighbor > i && coloring[neighbor] == -1) {
                is_candidate = 0;
                break;
            }
        }
        
        if (is_candidate) {
            unsigned int used_colors[MAX_COLOR_WORDS] = {0};
            
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                int nc = coloring[neighbor];
                if (nc >= 0 && nc < MAX_COLOR_WORDS * 32) {
                    used_colors[nc / 32] |= (1u << (nc % 32));
                }
            }
            
            int color = 0;
            for (int w = 0; w < MAX_COLOR_WORDS; w++) {
                if (~used_colors[w] != 0) {
                    color = w * 32 + __ffs(~used_colors[w]) - 1;
                    break;
                }
            }
            candidates[i] = color;
        }
    }
}


__device__ void reduce(int n, int *result) {
    int v = n;
    v += __shfl_down_sync(0xFFFFFFFF, v, 16);
    v += __shfl_down_sync(0xFFFFFFFF, v, 8);
    v += __shfl_down_sync(0xFFFFFFFF, v, 4);
    v += __shfl_down_sync(0xFFFFFFFF, v, 2);
    v += __shfl_down_sync(0xFFFFFFFF, v, 1);
    
    __shared__ int warp_sum[32];
    if (threadIdx.x % 32 == 0) {
        warp_sum[threadIdx.x / 32] = v;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        v = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sum[threadIdx.x] : 0;
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
    }
    if (threadIdx.x == 0) {
        result[blockIdx.x] = v;
    }

}

// Kernel to initialize random priorities for each vertex
__global__ void init_random_priorities(int *random_prio, int vertices, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int i = tid; i < vertices; i += gridsize) {
        // Use a simple hash-based PRNG for speed
        unsigned int h = seed ^ i;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        random_prio[i] = (int)h;
    }
}

// Hash function to generate per-color priority from base priority
__device__ __forceinline__ unsigned int hash_priority(int base_prio, int color) {
    unsigned int h = (unsigned int)base_prio ^ (color * 2654435761u);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    return h;
}

// ============================================================================
// Multi-color parallel MIS: Two-phase approach
// Phase 1: Each vertex picks the lowest partition it thinks it can join
// Phase 2: Resolve conflicts and commit colors
// ============================================================================
// PENDING_COLOR: Marker for vertices claimed but not yet colored
// This allows find and color phases to overlap
// ============================================================================
#define PENDING_COLOR (-2)

// Initialize per-vertex masks and partition assignment
// Initialize per-vertex candidate mask and partition assignment
__global__ void init_partition_masks(int vertices, int *coloring, int *partition_assignment,
                                     unsigned int *mask_candidate, int num_partitions) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    unsigned int all_mask = (num_partitions >= 32) ? 0xFFFFFFFFu : ((1u << num_partitions) - 1u);
    for (int i = tid; i < vertices; i += gridsize) {
        partition_assignment[i] = -1;
        if (coloring[i] >= 0 || coloring[i] == PENDING_COLOR) {
            mask_candidate[i] = 0u;
        } else {
            mask_candidate[i] = all_mask;
        }
    }
}

// Edge-parallel processing: each thread handles a range of adjacency indices
// Edge-parallel processing: clear candidate bits when a neighbor outranks this vertex
// Only consider edges where both endpoints are uncolored. Each bit p corresponds
// to one of the `num_partitions` parallel independent sets tried in this pass.
__global__ void process_edges_update_masks(int *vertex_pointers, int *adjacencies,
                                           int vertices, int edges,
                                           int *coloring,
                                           unsigned int *mask_candidate,
                                           int *random_prio, int num_partitions, int color_offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    for (int e = tid; e < edges; e += gridsize) {
        // find owner vertex v such that vertex_pointers[v] <= e < vertex_pointers[v+1]
        int lo = 0, hi = vertices - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (vertex_pointers[mid] <= e) lo = mid; else hi = mid - 1;
        }
        int v = lo;
        if (!(vertex_pointers[v] <= e && e < vertex_pointers[v + 1])) continue;

        int neighbor = adjacencies[e];

        // Only consider uncolored vertices; colored vertices are excluded from selection
        if (coloring[v] != -1) continue;
        // Skip neighbors that are already colored but not those pending
        if (coloring[neighbor] != -1 && coloring[neighbor] != PENDING_COLOR) continue;

        unsigned int clear_mask = 0u;
        for (int p = 0; p < num_partitions; ++p) {
            int actual_color = color_offset + p;
            unsigned int neigh_pr = hash_priority(random_prio[neighbor], actual_color);
            unsigned int my_pr = hash_priority(random_prio[v], actual_color);
            if (neigh_pr > my_pr || (neigh_pr == my_pr && neighbor > v)) {
                clear_mask |= (1u << p);
            }
        }
        if (clear_mask) atomicAnd(&mask_candidate[v], ~clear_mask);
    }
}

// Finalize per-vertex partition assignment from masks
__global__ void finalize_partition_assignment(int vertices, int *coloring, int *partition_assignment,
                                              unsigned int *mask_candidate,
                                              int num_partitions, int color_offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    for (int i = tid; i < vertices; i += gridsize) {
        if (coloring[i] >= 0 || coloring[i] == PENDING_COLOR) continue;
        unsigned int final_mask = mask_candidate[i];
        if (final_mask != 0u) {
            int bit = __ffs(final_mask) - 1;
            partition_assignment[i] = color_offset + bit;
            coloring[i] = PENDING_COLOR;  // Mark as pending
        } else {
            partition_assignment[i] = -1;
        }
    }
}


// Kernel 2: Conflict Resolution and Greedy Coloring for ONE partition (compact version)
// Only processes vertices from the compact list for this partition
// vertex_list contains the vertex IDs, list_size is the count
__global__ void multi_color_resolve_and_color_compact(
    int *vertex_pointers, int *adjacencies,
    int *coloring, int *partition_assignment, int *random_prio,
    int *vertex_list, int list_size, int target_partition) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    const int MAX_COLOR_WORDS = 16;  // Supports up to 256 colors
    
    for (int idx = tid; idx < list_size; idx += gridsize) {
        int i = vertex_list[idx];  // Get actual vertex ID from compact list
        
        // Double-check this vertex claimed the target partition
        if (partition_assignment[i] != target_partition) continue;
        
        unsigned int used_colors[MAX_COLOR_WORDS] = {0};
        
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            int nc = coloring[neighbor];
            // Only count real colors, not PENDING_COLOR
            if (nc >= 0 && nc < MAX_COLOR_WORDS * 32) {
                used_colors[nc / 32] |= (1u << (nc % 32));
            }
        }
        
        // Find first available color
        int color = 0;
        for (int w = 0; w < MAX_COLOR_WORDS; w++) {
            if (~used_colors[w] != 0) {
                color = w * 32 + __ffs(~used_colors[w]) - 1;
                break;
            }
        }
        
        coloring[i] = color;
    }
}

// Functor to check if vertex claimed any IS (partition_assignment >= color_offset)
struct claimed_any_is {
    int *partition_assignment;
    int color_offset;
    
    claimed_any_is(int *pa, int co) : partition_assignment(pa), color_offset(co) {}
    
    __host__ __device__ int operator()(int vertex_id) const {
        int p = partition_assignment[vertex_id];
        return (p >= color_offset) ? 1 : 0;
    }
};

// Kernel to scatter vertex indices into compact list using prefix sum positions
__global__ void scatter_to_compact_list(
    int *partition_assignment, int *prefix_sum, int *compact_list,
    int vertices, int color_offset) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int i = tid; i < vertices; i += gridsize) {
        int p = partition_assignment[i];
        if (p >= color_offset) {
            // This vertex claimed an IS, write to its position
            compact_list[prefix_sum[i]] = i;
        }
    }
}

// Legacy version that scans all vertices (kept for reference)
__global__ void multi_color_resolve_and_color(
    int *vertex_pointers, int *adjacencies, int vertices,
    int *coloring, int *partition_assignment, int *random_prio,
    int target_partition) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    const int MAX_COLOR_WORDS = 8;  // Supports up to 256 colors
    
    for (int i = tid; i < vertices; i += gridsize) {
        int my_partition = partition_assignment[i];
        
        // Only process vertices that claimed this specific partition
        if (my_partition != target_partition) continue;
        if (coloring[i] != -1) continue;   // Already colored (shouldn't happen)
        
        unsigned int my_prio = hash_priority(random_prio[i], my_partition);
        bool can_commit = true;
        
        // Check all neighbors who also claimed this partition
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            
            // If neighbor also claimed this partition, only one can win
            if (partition_assignment[neighbor] == my_partition) {
                unsigned int neighbor_prio = hash_priority(random_prio[neighbor], my_partition);
                // Lower priority loses, tie-break by vertex ID (lower ID wins)
                if (neighbor_prio > my_prio || (neighbor_prio == my_prio && neighbor < i)) {
                    can_commit = false;
                    break;
                }
            }
        }
        
        if (can_commit) {
            // We won the partition! Now find the smallest available color (greedy first-fit)
            // Since we process one partition at a time, all neighbors are either:
            // - Already colored from previous iterations/partitions (safe to read)
            // - In a different partition (not being colored now)
            // - Lost the conflict resolution (not being colored)
            unsigned int used_colors[MAX_COLOR_WORDS] = {0};
            
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                int nc = coloring[neighbor];
                if (nc >= 0 && nc < MAX_COLOR_WORDS * 32) {
                    used_colors[nc / 32] |= (1u << (nc % 32));
                }
            }
            
            // Find first available color
            int color = 0;
            for (int w = 0; w < MAX_COLOR_WORDS; w++) {
                if (~used_colors[w] != 0) {
                    color = w * 32 + __ffs(~used_colors[w]) - 1;
                    break;
                }
            }
            
            coloring[i] = color;
        }
    }
}

// Legacy single-kernel version (kept for reference)
__global__ void multi_color_mis_iter(
    int *vertex_pointers, int *adjacencies, int vertices,
    int *coloring, int *random_prio, int num_colors_to_try, int color_offset) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int i = tid; i < vertices; i += gridsize) {
        if (coloring[i] != -1) continue;  // Already colored
        
        int my_base_prio = random_prio[i];
        
        // Try each color in order, take the first one we can join
        for (int c = 0; c < num_colors_to_try; c++) {
            int actual_color = color_offset + c;
            unsigned int my_color_prio = hash_priority(my_base_prio, actual_color);
            
            // Check if this color is already used by a neighbor
            bool color_available = true;
            bool has_higher_priority_competitor = false;
            
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                int neighbor_color = coloring[neighbor];
                
                // If neighbor already has this color, we can't use it
                if (neighbor_color == actual_color) {
                    color_available = false;
                    break;
                }
                
                // If neighbor is uncolored, check if they have higher priority for this color
                if (neighbor_color == -1) {
                    unsigned int neighbor_color_prio = hash_priority(random_prio[neighbor], actual_color);
                    if (neighbor_color_prio > my_color_prio) {
                        has_higher_priority_competitor = true;
                        break;
                    }
                    // Tie-breaker: use vertex ID
                    if (neighbor_color_prio == my_color_prio && neighbor > i) {
                        has_higher_priority_competitor = true;
                        break;
                    }
                }
            }
            
            if (color_available && !has_higher_priority_competitor) {
                // We can take this color!
                coloring[i] = actual_color;
                break;  // Stop trying other colors
            }
        }
    }
}

// Randomized version: uses random priorities instead of vertex IDs
__global__ void luby_mis_coloring_iter_random(int *vertex_pointers, int *adjacencies, int vertices, 
                                              int *coloring, int *candidates, int *n_colored, 
                                              int *random_prio, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    const int MAX_COLOR_WORDS = 8;  // Supports up to 256 colors
    
    for (int i = tid; i < vertices; i += gridsize) {
        candidates[i] = -1;
        if (coloring[i] != -1) continue;
        
        int my_prio = random_prio[i];
        int is_candidate = 1;
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            if (random_prio[neighbor] > my_prio && coloring[neighbor] == -1) {
                is_candidate = 0;
                break;
            }
        }
        
        if (is_candidate) {
            unsigned int used_colors[MAX_COLOR_WORDS] = {0};
            
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                int nc = coloring[neighbor];
                if (nc >= 0 && nc < MAX_COLOR_WORDS * 32) {
                    used_colors[nc / 32] |= (1u << (nc % 32));
                }
            }
            
            int color = 0;
            for (int w = 0; w < MAX_COLOR_WORDS; w++) {
                if (~used_colors[w] != 0) {
                    color = w * 32 + __ffs(~used_colors[w]) - 1;
                    break;
                }
            }
            candidates[i] = color;
        }
    }
}

// Randomized version of apply_coloring
__global__ void apply_coloring_random(int *vertex_pointers, int *adjacencies, int vertices,
                                      int *coloring, int *candidates, int *n_colored, 
                                      int *random_prio, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int colored = 0;
    for (int i = tid; i < vertices; i += gridsize) {
        if (candidates[i] >= 0 && coloring[i] == -1) {
            int my_prio = random_prio[i];
            int safe = 1;
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                if (random_prio[neighbor] < my_prio && candidates[neighbor] >= 0) {
                    safe = 0;
                    break;
                }
            }
            if (safe) {
                coloring[i] = candidates[i];
                colored++;
            }
        }
    }
    __syncthreads();
    reduce(colored, n_colored);
}

__global__ void apply_coloring(int *vertex_pointers, int *adjacencies, int vertices,
                               int *coloring, int *candidates, int *n_colored, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int colored = 0;
    for (int i = tid; i < vertices; i += gridsize) {
        // Fix: check for >= 0 instead of truthiness
        if (candidates[i] >= 0 && coloring[i] == -1) {
            // Check no lower-ID neighbor is also a candidate
            int safe = 1;
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                if (neighbor < i && candidates[neighbor] >= 0) {
                    safe = 0;
                    break;
                }
            }
            if (safe) {
                coloring[i] = candidates[i];  // Use the precomputed color
                colored++;
            }
        }
    }
    __syncthreads();
    reduce(colored, n_colored);
}

__global__ void verify_coloring(int *vertex_pointers, int *adjacencies, int vertices, int *coloring,
    int *colors, int *uncolored_count, int *conflict_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;

    for (int i = tid; i < vertices; i += gridsize) {
        int my_color = coloring[i];
        if (my_color >= 0) {
            colors[my_color] = 1; //race condition not important
        }

        if (my_color == -1) {
            atomicAdd(uncolored_count, 1);
            continue;
        }
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            if (coloring[neighbor] == my_color) {
                atomicAdd(conflict_count, 1);
                break;
            }
        }
    }
}


// Compute total active edges across all active vertices
// Uses the reduce function to do block-level reduction, final reduction on CPU
__global__ void compute_active_edges(int *vertex_pointers, int *adjacencies, int vertices, 
                                     int *priorities, int *block_edge_sums) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int local_edges = 0;
    
    for (int i = tid; i < vertices; i += gridsize) {
        if (priorities[i] == -1) { // active vertex
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                if (priorities[neighbor] == -1) {
                    local_edges++;
                }
            }
        }
    }
    
    __syncthreads();
    reduce(local_edges, block_edge_sums);
}


//at each iteration, we would like to color the vertices which have maximum degree
__global__ void smallest_last_ordering(int *vertex_pointers, int *adjacencies, int vertices, int activeVertices, int iteration, int *priorities, int *degrees, int average_degree, float eps, int *n_marked) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    int threshold = average_degree * (1 + eps);

    int marked_count = 0;

    for (int i = tid; i < vertices; i += gridsize) {
        if (priorities[i] == -1) { //active vertex
            if (degrees[i] <= threshold) {
                priorities[i] = iteration; //mark as processed
                marked_count++;
            }
        }
    }
    __syncthreads();
    reduce(marked_count, n_marked);
}

__global__ void update_degrees(int *vertex_pointers, int *adjacencies, int vertices, int iteration, int *priorities, int *degrees) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    for (int i = tid; i < vertices; i += gridsize) {
        if (priorities[i] == -1) { //active vertex
            int diff = 0;
            //scan over neighbors to decrement degree for marked neighbors
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                if (priorities[neighbor] == iteration) {
                    diff++;
                }
            }
            degrees[i] -= diff;
        }
    }

}


// ============================================================================
// Sort-based graph reordering by priority
// Reorders vertices so same-priority vertices are contiguous
// Single graph, single allocation - just track priority boundaries
// ============================================================================

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

// Scatter edges using the permutation (keeps ALL edges, just remaps indices)
// perm: new_idx -> old_idx (which vertex goes where)
// inv_perm: old_idx -> new_idx (where each vertex went)
__global__ void scatter_edges_by_perm(
    int *old_vertex_pointers, int *old_adjacencies, int vertices,
    int *perm, int *inv_perm,
    int *new_vertex_pointers, int *new_adjacencies,
    int *edge_offsets) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int new_idx = tid; new_idx < vertices; new_idx += gridsize) {
        int old_idx = perm[new_idx];
        int degree = old_vertex_pointers[old_idx + 1] - old_vertex_pointers[old_idx];
        int out_start = edge_offsets[new_idx];
        
        new_vertex_pointers[new_idx] = out_start;
        
        for (int j = 0; j < degree; j++) {
            int old_neighbor = old_adjacencies[old_vertex_pointers[old_idx] + j];
            // Remap neighbor through inverse permutation
            new_adjacencies[out_start + j] = inv_perm[old_neighbor];
        }
    }
}

// Result structure for sorted graph
struct SortedGraph {
    int *vertex_pointers;    // CSC pointers for reordered graph
    int *adjacencies;        // CSC adjacencies (remapped indices)
    int *perm;               // new_idx -> old_idx
    int *inv_perm;           // old_idx -> new_idx  
    int *priority_offsets;   // priority p vertices are in [priority_offsets[p], priority_offsets[p+1])
    int num_priorities;
    int vertices;
    int edges;
};

// Sort graph by priority - creates a reordered graph where same-priority vertices are contiguous
// Returns pointers to device memory (caller responsible for freeing)
void sort_graph_by_priority(
    int *d_vertex_pointers, int *d_adjacencies, int vertices, int edges,
    int *d_priorities, int num_priorities,
    SortedGraph &result,
    int numBlocks, int blockSize) {
    
    result.vertices = vertices;
    result.edges = edges;  // Same number of edges - we keep them all
    result.num_priorities = num_priorities;
    
    // Allocate permutation arrays
    int *d_perm, *d_inv_perm, *d_priorities_copy;
    cudaMalloc(&d_perm, sizeof(int) * vertices);
    cudaMalloc(&d_inv_perm, sizeof(int) * vertices);
    cudaMalloc(&d_priorities_copy, sizeof(int) * vertices);
    
    // Initialize perm to identity: perm[i] = i
    thrust::device_ptr<int> perm_ptr(d_perm);
    thrust::sequence(perm_ptr, perm_ptr + vertices);
    
    // Copy priorities (sort will modify keys)
    cudaMemcpy(d_priorities_copy, d_priorities, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);
    
    // Sort perm by priorities: after this, perm[new_idx] = old_idx
    // Vertices are grouped by priority
    thrust::device_ptr<int> prio_ptr(d_priorities_copy);
    thrust::sort_by_key(prio_ptr, prio_ptr + vertices, perm_ptr);
    
    // Build inverse permutation: inv_perm[old_idx] = new_idx
    thrust::device_ptr<int> inv_perm_ptr(d_inv_perm);
    thrust::scatter(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(vertices),
        perm_ptr,
        inv_perm_ptr);
    
    // Compute priority boundaries by scanning sorted priorities
    int *h_priority_offsets = (int*)malloc(sizeof(int) * (num_priorities + 1));
    thrust::host_vector<int> h_sorted_prios(vertices);
    cudaMemcpy(h_sorted_prios.data(), d_priorities_copy, sizeof(int) * vertices, cudaMemcpyDeviceToHost);
    
    h_priority_offsets[0] = 0;
    int current_prio = 0;
    for (int i = 0; i < vertices && current_prio < num_priorities; i++) {
        while (current_prio < h_sorted_prios[i] && current_prio < num_priorities) {
            current_prio++;
            h_priority_offsets[current_prio] = i;
        }
    }
    while (current_prio < num_priorities) {
        current_prio++;
        h_priority_offsets[current_prio] = vertices;
    }
    h_priority_offsets[num_priorities] = vertices;
    
    cudaMalloc(&result.priority_offsets, sizeof(int) * (num_priorities + 1));
    cudaMemcpy(result.priority_offsets, h_priority_offsets, sizeof(int) * (num_priorities + 1), cudaMemcpyHostToDevice);
    
    // Gather degrees into new order and compute edge offsets
    // reordered_degrees[new_idx] = degree of perm[new_idx] in original graph
    int *d_reordered_degrees, *d_edge_offsets;
    cudaMalloc(&d_reordered_degrees, sizeof(int) * vertices);
    cudaMalloc(&d_edge_offsets, sizeof(int) * vertices);
    
    // Compute degrees by gathering from vertex_pointers differences
    // degree[old] = vertex_pointers[old+1] - vertex_pointers[old]
    // We need reordered_degrees[new] = degree[perm[new]]
    thrust::device_ptr<int> vp_ptr(d_vertex_pointers);
    thrust::device_ptr<int> reord_deg_ptr(d_reordered_degrees);
    
    // Use transform with gather to compute reordered degrees
    thrust::transform(
        thrust::make_permutation_iterator(vp_ptr + 1, perm_ptr),
        thrust::make_permutation_iterator(vp_ptr + 1, perm_ptr + vertices),
        thrust::make_permutation_iterator(vp_ptr, perm_ptr),
        reord_deg_ptr,
        thrust::minus<int>());
    
    // Prefix sum to get edge offsets
    thrust::device_ptr<int> eo_ptr(d_edge_offsets);
    thrust::exclusive_scan(reord_deg_ptr, reord_deg_ptr + vertices, eo_ptr);
    
    // Allocate output graph (same size as input - all edges kept)
    cudaMalloc(&result.vertex_pointers, sizeof(int) * (vertices + 1));
    cudaMalloc(&result.adjacencies, sizeof(int) * edges);
    
    // Scatter edges
    scatter_edges_by_perm<<<numBlocks, blockSize>>>(
        d_vertex_pointers, d_adjacencies, vertices,
        d_perm, d_inv_perm,
        result.vertex_pointers, result.adjacencies,
        d_edge_offsets);
    
    // Set final vertex pointer
    thrust::device_ptr<int> new_vp_ptr(result.vertex_pointers);
    new_vp_ptr[vertices] = edges;
    
    // Store permutations in result
    result.perm = d_perm;
    result.inv_perm = d_inv_perm;
    
    // Cleanup temporaries
    cudaFree(d_priorities_copy);
    cudaFree(d_reordered_degrees);
    cudaFree(d_edge_offsets);
    free(h_priority_offsets);
}

// Free sorted graph
void free_sorted_graph(SortedGraph &g) {
    if (g.vertex_pointers) cudaFree(g.vertex_pointers);
    if (g.adjacencies) cudaFree(g.adjacencies);
    if (g.perm) cudaFree(g.perm);
    if (g.inv_perm) cudaFree(g.inv_perm);
    if (g.priority_offsets) cudaFree(g.priority_offsets);
}


// Map coloring from sorted order back to original vertex order
__global__ void unmap_coloring(
    int *sorted_coloring,   // coloring in sorted vertex order
    int *original_coloring, // output: coloring in original vertex order
    int *perm,              // perm[sorted_idx] = original_idx
    int vertices) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int sorted_idx = tid; sorted_idx < vertices; sorted_idx += gridsize) {
        int original_idx = perm[sorted_idx];
        original_coloring[original_idx] = sorted_coloring[sorted_idx];
    }
}

// Functor to check if a vertex is active (priority == -1)
struct is_active {
    __host__ __device__ bool operator()(int priority) const {
        return priority == -1;
    }
};

// Functor to check if a vertex is colored (color >= 0)
struct is_colored {
    __host__ __device__ bool operator()(int color) const {
        return color >= 0;
    }
};

// Functor to compute active edges for a vertex
struct active_edge_counter {
    int *vertex_pointers;
    int *adjacencies;
    int *priorities;
    
    active_edge_counter(int *vp, int *adj, int *prio) 
        : vertex_pointers(vp), adjacencies(adj), priorities(prio) {}
    
    __host__ __device__ int operator()(int i) const {
        if (priorities[i] != -1) return 0;
        int count = 0;
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            if (priorities[adjacencies[j]] == -1) {
                count++;
            }
        }
        return count;
    }
};

// Host function to get average degree across active nodes using Thrust
// Returns average degree (total_active_edges / active_vertices)
float get_average_degree(int *d_vertex_pointers, int *d_adjacencies, int vertices, int *d_priorities) {
    thrust::device_ptr<int> prio_ptr(d_priorities);
    
    // Count active vertices using Thrust
    int total_vertices = thrust::count_if(prio_ptr, prio_ptr + vertices, is_active());
    
    if (total_vertices == 0) return 0.0f;
    
    // Count active edges using transform_reduce
    active_edge_counter counter(d_vertex_pointers, d_adjacencies, d_priorities);
    int total_edges = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(vertices),
        counter,
        0,
        thrust::plus<int>());
    
    return (float)total_edges / (float)total_vertices;
}

// Legacy kernel-based versions kept for compatibility
// Compute count of active vertices
// Uses the reduce function to do block-level reduction, final reduction on CPU
__global__ void compute_active_vertices(int *priorities, int vertices, int *block_vertex_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int local_vertices = 0;
    
    for (int i = tid; i < vertices; i += gridsize) {
        if (priorities[i] == -1) { // active vertex
            local_vertices++;
        }
    }
    
    __syncthreads();
    reduce(local_vertices, block_vertex_counts);
}

int main(int argc, char *argv[]) {
    int *h_adjacencies = nullptr;
    int *h_vertex_pointers = nullptr;
    int edges = 0;
    int vertices = 0;
    
    // Parse command line arguments
    bool useSorting = true;  // Default: use sorting optimization
    bool useRandom = false;  // Default: use vertex ID comparison
    bool useMultiColor = false;  // Default: use single-color MIS per iteration
    int numParallelColors = 8;   // Number of colors to try in parallel when --multi-color
    // Build CSC from edge list
    const char *edgeFile = (argc > 1) ? argv[1] : "1684.edges";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-sort") == 0) {
            useSorting = false;
        } else if (strcmp(argv[i], "--sort") == 0) {
            useSorting = true;
        } else if (strcmp(argv[i], "--random") == 0) {
            useRandom = true;
        } else if (strcmp(argv[i], "--no-random") == 0) {
            useRandom = false;
        } else if (strcmp(argv[i], "--multi-color") == 0) {
            useMultiColor = true;
            useRandom = true;  // Multi-color requires random priorities
        } else if (strncmp(argv[i], "--parallel-colors=", 18) == 0) {
            numParallelColors = atoi(argv[i] + 18);
            if (numParallelColors < 1) numParallelColors = 8;
        } else {
            edgeFile = argv[i];
        }
    }
    
    printf("[INFO] Sorting optimization: %s\n", useSorting ? "ENABLED" : "DISABLED");
    printf("[INFO] Random priorities: %s\n", useRandom ? "ENABLED" : "DISABLED");
    printf("[INFO] Multi-color parallel MIS: %s\n", useMultiColor ? "ENABLED" : "DISABLED");
    if (useMultiColor) {
        printf("[INFO] Parallel colors per iteration: %d\n", numParallelColors);
    }
    

    std::vector<int> colPtr, rowIdx;
    // Assume input is 0-based; set false if file is 1-based
    // Detect CSV format by file extension
    bool isCsv = false;
    std::string edgeFileStr(edgeFile);
    if (edgeFileStr.size() >= 4 && edgeFileStr.substr(edgeFileStr.size() - 4) == ".csv") {
        isCsv = true;
    }
    printf("[INFO] Parsing edges from: %s (CSV mode: %s)\n", edgeFile, isCsv ? "yes" : "no");
    load_edges_to_csc(edgeFile, colPtr, rowIdx, vertices, edges, true, isCsv);
    if (vertices == 0 || edges == 0) {
        fprintf(stderr, "No edges/vertices parsed from %s. Exiting.\n", edgeFile);
        return 1;
    }
    printf("[INFO] Parsed graph: vertices=%d, edges=%d\n", vertices, edges);
    // Allocate host arrays and copy from vectors
    h_vertex_pointers = (int*)malloc(sizeof(int) * (vertices + 1));
    h_adjacencies = (int*)malloc(sizeof(int) * edges);
    std::copy(colPtr.begin(), colPtr.end(), h_vertex_pointers);
    std::copy(rowIdx.begin(), rowIdx.end(), h_adjacencies);
    printf("[INFO] Built CSC: colPtr size=%d, rowIdx size=%d\n", vertices + 1, edges);
    //to get a list of vertices adjacent to vertex i, use adjncy[xadj[i]] to adjncy[xadj[i+1]-1]
    int *adjacencies, *vertex_pointers, *coloring;
    int *n_colored;
    int *candidates;

    
    int blockSize = 256;
    int numBlocks = 160;

    printf("[INFO] Allocating device memory...\n");
    cudaMalloc((void **)&vertex_pointers, sizeof(int) * (vertices + 1));
    cudaMalloc((void **)&adjacencies, sizeof(int) * edges);
    cudaMalloc((void **)&coloring, sizeof(int) * vertices);
    cudaMalloc((void **)&n_colored, sizeof(int) * numBlocks);
    cudaMalloc((void **)&candidates, sizeof(int) * vertices);

    //init to zero
    cudaMemset(n_colored, 0, sizeof(int) * numBlocks);
    cudaMemcpy(vertex_pointers, h_vertex_pointers, sizeof(int) * (vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(adjacencies, h_adjacencies, sizeof(int) * edges, cudaMemcpyHostToDevice);
    printf("[INFO] Host-to-device copy complete.\n");

    int *colors;
    cudaMalloc((void **)&colors, sizeof(int) * (vertices+1));
    cudaMemset(colors, 0, sizeof(int) * (vertices+1));

    // Variables for sorting optimization (may or may not be used)
    int *d_priorities = nullptr, *d_degrees = nullptr, *d_n_marked = nullptr;
    SortedGraph sorted_graph = {nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, 0};
    int *sorted_coloring = nullptr, *sorted_candidates = nullptr;

    // Random priority array (used when --random is enabled)
    int *d_random_prio = nullptr;
    if (useRandom) {
        cudaMalloc(&d_random_prio, sizeof(int) * vertices);
        // Initialize with random values (use time as seed for variety)
        init_random_priorities<<<numBlocks, blockSize>>>(d_random_prio, vertices, (unsigned int)time(nullptr));
        cudaDeviceSynchronize();
        printf("[INFO] Random priorities initialized.\n");
    }

    // Start total timer
    cudaEvent_t total_start, total_stop;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventRecord(total_start);

    if (useSorting) {
        // ========================================================================
        // Step 1: Compute degrees for sorting
        // ========================================================================
        printf("[INFO] Computing degrees for degree-based sorting...\n");
        
        cudaMalloc(&d_priorities, sizeof(int) * vertices);
        cudaMalloc(&d_degrees, sizeof(int) * vertices);
        
        // Compute degrees from vertex_pointers (degree = vertex_pointers[i+1] - vertex_pointers[i])
        thrust::device_ptr<int> vp_ptr(vertex_pointers);
        thrust::device_ptr<int> deg_ptr(d_degrees);
        thrust::transform(vp_ptr + 1, vp_ptr + vertices + 1, vp_ptr, deg_ptr, thrust::minus<int>());
        
        // Copy degrees to priorities (we'll sort by degree)
        cudaMemcpy(d_priorities, d_degrees, sizeof(int) * vertices, cudaMemcpyDeviceToDevice);
        
        // Find max degree to determine number of partitions
        int max_degree = thrust::reduce(deg_ptr, deg_ptr + vertices, 0, thrust::maximum<int>());
        int num_partitions = max_degree + 1;
        
        cudaEvent_t deg_start, deg_stop;
        cudaEventCreate(&deg_start);
        cudaEventCreate(&deg_stop);
        cudaEventRecord(deg_start);
        cudaEventRecord(deg_stop);
        cudaEventSynchronize(deg_stop);
        float deg_ms = 0;
        cudaEventElapsedTime(&deg_ms, deg_start, deg_stop);
        printf("[INFO] Degree computation completed in %f ms, max degree=%d, %d partitions\n", deg_ms, max_degree, num_partitions);
        
        // ========================================================================
        // Step 2: Sort graph by partition
        // ========================================================================
        printf("[INFO] Sorting graph by partition...\n");
        
        cudaEvent_t sort_start, sort_stop;
        cudaEventCreate(&sort_start);
        cudaEventCreate(&sort_stop);
        cudaEventRecord(sort_start);
        
        sort_graph_by_priority(vertex_pointers, adjacencies, vertices, edges,
                               d_priorities, num_partitions, sorted_graph, numBlocks, blockSize);
        
        cudaEventRecord(sort_stop);
        cudaEventSynchronize(sort_stop);
        float sort_ms = 0;
        cudaEventElapsedTime(&sort_ms, sort_start, sort_stop);
        printf("[INFO] Graph sorted in %f ms\n", sort_ms);
        
        // ========================================================================
        // Step 3: Run Luby coloring on sorted graph
        // ========================================================================
        cudaMalloc(&sorted_coloring, sizeof(int) * vertices);
        cudaMalloc(&sorted_candidates, sizeof(int) * vertices);
        cudaMemset(sorted_coloring, -1, sizeof(int) * vertices);
        
        int cum_colored = 0;
        float total_luby_ms = 0, total_apply_ms = 0;
        int color_offset = 0;
        
        if (useMultiColor) {
            printf("[INFO] Starting multi-color parallel MIS on sorted graph (%d IS/iter, 2-stream pipeline)...\n", numParallelColors);
            
            // Double-buffered arrays for pipelining
            int *d_partition_assign[2];
            int *d_flags[2];
            int *d_prefix_sum[2];
            int *d_compact_list[2];
            unsigned int *d_mask_candidate[2];
            int total_claimed[2] = {0, 0};
            
            for (int b = 0; b < 2; b++) {
                cudaMalloc(&d_partition_assign[b], sizeof(int) * vertices);
                cudaMalloc(&d_flags[b], sizeof(int) * vertices);
                cudaMalloc(&d_prefix_sum[b], sizeof(int) * vertices);
                cudaMalloc(&d_compact_list[b], sizeof(int) * vertices);
                cudaMalloc(&d_mask_candidate[b], sizeof(unsigned int) * vertices);
            }
            
            // Vertex indices for transform
            int *d_vertex_indices;
            cudaMalloc(&d_vertex_indices, sizeof(int) * vertices);
            thrust::device_ptr<int> idx_ptr(d_vertex_indices);
            thrust::sequence(idx_ptr, idx_ptr + vertices);
            
            // Two streams: one for finding IS, one for coloring
            cudaStream_t stream_find, stream_color;
            cudaStreamCreate(&stream_find);
            cudaStreamCreate(&stream_color);
            
            // Events for synchronization
            cudaEvent_t find_done[2], mark_done[2];
            cudaEventCreate(&find_done[0]);
            cudaEventCreate(&find_done[1]);
            cudaEventCreate(&mark_done[0]);
            cudaEventCreate(&mark_done[1]);
            
            int iter = 0;
            int buf = 0;
            int color_offset_find = 0;  // color_offset for finding
            int color_offset_color = 0; // color_offset for coloring
            
            // Timing for wait overhead
            float total_wait_find_ms = 0;   // Time stream_color waits for stream_find
            float total_wait_mark_ms = 0;   // Time stream_find waits for mark_done
            float total_wait_color_ms = 0;  // Time waiting for stream_color to sync
            
            // Kick off first IS finding
            init_partition_masks<<<numBlocks, blockSize, 0, stream_find>>>(
                vertices, sorted_coloring, d_partition_assign[buf], d_mask_candidate[buf], numParallelColors);
            // Process edges in parallel to update masks
            process_edges_update_masks<<<numBlocks, blockSize, 0, stream_find>>>(
                sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, sorted_graph.edges,
                sorted_coloring, d_mask_candidate[buf], d_random_prio, numParallelColors, color_offset_find);
            // Finalize assignment
            finalize_partition_assignment<<<numBlocks, blockSize, 0, stream_find>>>(
                vertices, sorted_coloring, d_partition_assign[buf], d_mask_candidate[buf], numParallelColors, color_offset_find);
            
            // Build flags, prefix sum, scatter (still on stream_find)
            cudaStreamSynchronize(stream_find);
            thrust::device_ptr<int> flags_ptr(d_flags[buf]);
            thrust::device_ptr<int> prefix_ptr(d_prefix_sum[buf]);
            claimed_any_is pred(d_partition_assign[buf], color_offset_find);
            thrust::transform(idx_ptr, idx_ptr + vertices, flags_ptr, pred);
            thrust::exclusive_scan(flags_ptr, flags_ptr + vertices, prefix_ptr);
            
            int last_prefix, last_flag;
            cudaMemcpy(&last_prefix, d_prefix_sum[buf] + vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_flag, d_flags[buf] + vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
            total_claimed[buf] = last_prefix + last_flag;
            
            if (total_claimed[buf] > 0) {
                scatter_to_compact_list<<<numBlocks, blockSize, 0, stream_find>>>(
                    d_partition_assign[buf], d_prefix_sum[buf], d_compact_list[buf],
                    vertices, color_offset_find);
            }
            cudaEventRecord(find_done[buf], stream_find);
            cudaEventRecord(mark_done[buf], stream_find);
            color_offset_find += numParallelColors;
            
            while (cum_colored < vertices) {
                cudaEvent_t iter_start, iter_stop;
                cudaEventCreate(&iter_start);
                cudaEventCreate(&iter_stop);
                cudaEventRecord(iter_start);
                
                int next_buf = 1 - buf;
                color_offset_color = color_offset_find - numParallelColors;
                
                // Stream_color: Wait for current IS finding to complete, then color
                // Measure how long we wait for find to complete
                cudaEvent_t wait_find_start, wait_find_end;
                cudaEventCreate(&wait_find_start);
                cudaEventCreate(&wait_find_end);
                cudaEventRecord(wait_find_start, stream_color);
                cudaStreamWaitEvent(stream_color, find_done[buf], 0);
                cudaEventRecord(wait_find_end, stream_color);
                
                if (total_claimed[buf] > 0) {
                    for (int p = 0; p < numParallelColors; p++) {
                        int target_partition = color_offset_color + p;
                        int colorBlocks = min(numBlocks, (total_claimed[buf] + blockSize - 1) / blockSize);
                        multi_color_resolve_and_color_compact<<<colorBlocks, blockSize, 0, stream_color>>>(
                            sorted_graph.vertex_pointers, sorted_graph.adjacencies,
                            sorted_coloring, d_partition_assign[buf], d_random_prio,
                            d_compact_list[buf], total_claimed[buf], target_partition);
                    }
                }
                
                // Stream_find: Wait for mark_pending to complete (not coloring!), then find next IS
                // Measure how long we wait for mark to complete
                cudaEvent_t wait_mark_start, wait_mark_end;
                cudaEventCreate(&wait_mark_start);
                cudaEventCreate(&wait_mark_end);
                cudaEventRecord(wait_mark_start, stream_find);
                cudaStreamWaitEvent(stream_find, mark_done[buf], 0);
                cudaEventRecord(wait_mark_end, stream_find);
                
                init_partition_masks<<<numBlocks, blockSize, 0, stream_find>>>(
                    vertices, sorted_coloring, d_partition_assign[next_buf], d_mask_candidate[next_buf], numParallelColors);
                process_edges_update_masks<<<numBlocks, blockSize, 0, stream_find>>>(
                    sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, sorted_graph.edges,
                    sorted_coloring, d_mask_candidate[next_buf], d_random_prio, numParallelColors, color_offset_find);
                finalize_partition_assignment<<<numBlocks, blockSize, 0, stream_find>>>(
                    vertices, sorted_coloring, d_partition_assign[next_buf], d_mask_candidate[next_buf], numParallelColors, color_offset_find);
                
                // Synchronize stream_find for thrust operations
                cudaStreamSynchronize(stream_find);
                
                thrust::device_ptr<int> next_flags_ptr(d_flags[next_buf]);
                thrust::device_ptr<int> next_prefix_ptr(d_prefix_sum[next_buf]);
                claimed_any_is next_pred(d_partition_assign[next_buf], color_offset_find);
                thrust::transform(idx_ptr, idx_ptr + vertices, next_flags_ptr, next_pred);
                thrust::exclusive_scan(next_flags_ptr, next_flags_ptr + vertices, next_prefix_ptr);
                
                cudaMemcpy(&last_prefix, d_prefix_sum[next_buf] + vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&last_flag, d_flags[next_buf] + vertices - 1, sizeof(int), cudaMemcpyDeviceToHost);
                total_claimed[next_buf] = last_prefix + last_flag;
                
                if (total_claimed[next_buf] > 0) {
                    scatter_to_compact_list<<<numBlocks, blockSize, 0, stream_find>>>(
                        d_partition_assign[next_buf], d_prefix_sum[next_buf], d_compact_list[next_buf],
                        vertices, color_offset_find);
                }
                cudaEventRecord(find_done[next_buf], stream_find);
                cudaEventRecord(mark_done[next_buf], stream_find);
                color_offset_find += numParallelColors;
                
                // Wait for coloring to count results - measure wait time
                cudaEvent_t wait_color_start, wait_color_end;
                cudaEventCreate(&wait_color_start);
                cudaEventCreate(&wait_color_end);
                cudaEventRecord(wait_color_start);  // default stream
                cudaStreamSynchronize(stream_color);
                cudaEventRecord(wait_color_end);
                cudaEventSynchronize(wait_color_end);
                
                // Accumulate wait times
                float wait_find_ms = 0, wait_mark_ms = 0, wait_color_ms = 0;
                cudaEventElapsedTime(&wait_find_ms, wait_find_start, wait_find_end);
                cudaEventElapsedTime(&wait_mark_ms, wait_mark_start, wait_mark_end);
                cudaEventElapsedTime(&wait_color_ms, wait_color_start, wait_color_end);
                total_wait_find_ms += wait_find_ms;
                total_wait_mark_ms += wait_mark_ms;
                total_wait_color_ms += wait_color_ms;
                
                cudaEventDestroy(wait_find_start);
                cudaEventDestroy(wait_find_end);
                cudaEventDestroy(wait_mark_start);
                cudaEventDestroy(wait_mark_end);
                cudaEventDestroy(wait_color_start);
                cudaEventDestroy(wait_color_end);
                
                cudaEventRecord(iter_stop);
                cudaEventSynchronize(iter_stop);
                float iter_ms = 0;
                cudaEventElapsedTime(&iter_ms, iter_start, iter_stop);
                total_luby_ms += iter_ms;
                
                // Count colored vertices
                thrust::device_ptr<int> col_ptr(sorted_coloring);
                int new_total = thrust::count_if(col_ptr, col_ptr + vertices, is_colored());
                int new_colored = new_total - cum_colored;
                cum_colored = new_total;
                
                printf("[INFO] Iteration %d: time=%f ms, claimed=%d, colored=%d, cumulative=%d/%d (wait: find=%.3f, mark=%.3f, color=%.3f ms)\n", 
                       iter, iter_ms, total_claimed[buf], new_colored, cum_colored, vertices,
                       wait_find_ms, wait_mark_ms, wait_color_ms);
                
                buf = next_buf;
                iter++;
                
                // Safety check
                if (new_colored == 0 && cum_colored < vertices) {
                    printf("[WARN] No progress made, breaking...\n");
                    break;
                }
                
                cudaEventDestroy(iter_start);
                cudaEventDestroy(iter_stop);
            }
            
            // Print wait time summary
            printf("[INFO] Total wait times: find=%.3f ms, mark=%.3f ms, color=%.3f ms\n",
                   total_wait_find_ms, total_wait_mark_ms, total_wait_color_ms);
            printf("[INFO] Overlap efficiency: %.2f%% (ideal: wait times near 0)\n",
                   100.0 * (1.0 - (total_wait_find_ms + total_wait_mark_ms) / total_luby_ms));
            
            // Cleanup
            cudaStreamDestroy(stream_find);
            cudaStreamDestroy(stream_color);
            cudaEventDestroy(find_done[0]);
            cudaEventDestroy(find_done[1]);
            cudaEventDestroy(mark_done[0]);
            cudaEventDestroy(mark_done[1]);
            for (int b = 0; b < 2; b++) {
                cudaFree(d_partition_assign[b]);
                cudaFree(d_flags[b]);
                cudaFree(d_prefix_sum[b]);
                cudaFree(d_compact_list[b]);
                cudaFree(d_mask_candidate[b]);
            }
            cudaFree(d_vertex_indices);
            
        } else {
            printf("[INFO] Starting Luby MIS coloring iterations on sorted graph...\n");
            int iter = 0;

            while (cum_colored < vertices) {
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
                
                if (useRandom) {
                    luby_mis_coloring_iter_random<<<numBlocks, blockSize>>>(
                        sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, 
                        sorted_coloring, sorted_candidates, n_colored, d_random_prio, iter);
                    cudaDeviceSynchronize();
                    apply_coloring_random<<<numBlocks, blockSize>>>(
                        sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, 
                        sorted_coloring, sorted_candidates, n_colored, d_random_prio, iter);
                } else {
                    luby_mis_coloring_iter<<<numBlocks, blockSize>>>(
                        sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, 
                        sorted_coloring, sorted_candidates, n_colored, iter);
                    cudaDeviceSynchronize();
                    apply_coloring<<<numBlocks, blockSize>>>(
                        sorted_graph.vertex_pointers, sorted_graph.adjacencies, vertices, 
                        sorted_coloring, sorted_candidates, n_colored, iter);
                }
                
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float iter_ms = 0;
                cudaEventElapsedTime(&iter_ms, start, stop);
                total_luby_ms += iter_ms;
                
                // Count colored vertices using thrust
                thrust::device_ptr<int> col_ptr(sorted_coloring);
                int new_total = thrust::count_if(col_ptr, col_ptr + vertices, is_colored());
                int new_colored = new_total - cum_colored;
                cum_colored = new_total;
                
                printf("[INFO] Iteration %d: time=%f ms, colored=%d, cumulative=%d/%d\n", 
                       iter, iter_ms, new_colored, cum_colored, vertices);
                iter++;
                
                // Safety check to prevent infinite loop
                if (new_colored == 0 && cum_colored < vertices) {
                    printf("[WARN] No progress made, forcing remaining vertices...\n");
                    break;
                }
            }
        }
        
        printf("[INFO] Total coloring kernel time: %f ms\n", total_luby_ms);
        
        // ========================================================================
        // Step 4: Unmap coloring back to original vertex order
        // ========================================================================
        printf("[INFO] Unmapping coloring to original vertex order...\n");
        unmap_coloring<<<numBlocks, blockSize>>>(sorted_coloring, coloring, sorted_graph.perm, vertices);
        cudaDeviceSynchronize();
    } else {
        // ========================================================================
        // No sorting - run Luby coloring directly on original graph
        // ========================================================================
        cudaMemset(coloring, -1, sizeof(int) * vertices);
        int cum_colored = 0;
        float total_luby_ms = 0, total_apply_ms = 0;
        printf("[INFO] Starting Luby MIS coloring iterations (no sorting)...\n");
        int iter = 0;

        while (cum_colored < vertices) {
            cudaEvent_t start, stop, mid;
            cudaEventCreate(&start);
            cudaEventCreate(&mid);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            if (useRandom) {
                luby_mis_coloring_iter_random<<<numBlocks, blockSize>>>(
                    vertex_pointers, adjacencies, vertices, 
                    coloring, candidates, n_colored, d_random_prio, iter);
            } else {
                luby_mis_coloring_iter<<<numBlocks, blockSize>>>(
                    vertex_pointers, adjacencies, vertices, 
                    coloring, candidates, n_colored, iter);
            }
            
            cudaEventRecord(mid);
            cudaEventSynchronize(mid);
            
            if (useRandom) {
                apply_coloring_random<<<numBlocks, blockSize>>>(
                    vertex_pointers, adjacencies, vertices, 
                    coloring, candidates, n_colored, d_random_prio, iter);
            } else {
                apply_coloring<<<numBlocks, blockSize>>>(
                    vertex_pointers, adjacencies, vertices, 
                    coloring, candidates, n_colored, iter);
            }
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float luby_ms = 0, apply_ms = 0;
            cudaEventElapsedTime(&luby_ms, start, mid);
            cudaEventElapsedTime(&apply_ms, mid, stop);
            total_luby_ms += luby_ms;
            total_apply_ms += apply_ms;
            printf("[INFO] Iteration %d: luby_mis_coloring_iter=%f ms, apply_coloring=%f ms\n", iter, luby_ms, apply_ms);

            int h_n_colored[numBlocks];
            int new_colored = 0;
            cudaMemcpy(h_n_colored, n_colored, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
            for (int i = 0; i < numBlocks; i++) {
                new_colored += h_n_colored[i];
            }
            cum_colored += new_colored;
            printf("[INFO] Iteration %d colored=%d, cumulative colored=%d/%d\n", iter, new_colored, cum_colored, vertices);
            iter++;
        }
        
        printf("[INFO] Total luby_mis_coloring_iter time: %f ms\n", total_luby_ms);
        printf("[INFO] Total apply_coloring time: %f ms\n", total_apply_ms);
    }
    
    // Stop total timer and print
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, total_start, total_stop);
    printf("[INFO] Coloring complete.\n");
    printf("[INFO] Total coloring time: %f ms\n", total_ms);
    
    cudaDeviceSynchronize();
    int h_uncolored_count = 0;
    int h_conflict_count = 0;
    int *uncolored_count, *conflict_count;
    cudaMalloc((void **)&uncolored_count, sizeof(int));
    cudaMalloc((void **)&conflict_count, sizeof(int));
    cudaMemcpy(uncolored_count, &h_uncolored_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(conflict_count, &h_conflict_count, sizeof(int), cudaMemcpyHostToDevice);
    printf("[INFO] Verifying coloring...\n");
    verify_coloring<<<numBlocks, blockSize>>>(vertex_pointers, adjacencies, vertices, coloring,
        colors, uncolored_count,  conflict_count);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_uncolored_count, uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_conflict_count, conflict_count, sizeof(int), cudaMemcpyDeviceToHost); 
    printf("Total vertices: %d\n", vertices);
    printf("Uncolored vertices: %d\n", h_uncolored_count);
    printf("Conflicting vertices: %d\n", h_conflict_count);
    // Count total number of colors used
    std::vector<int> h_colors(vertices + 1);
    cudaMemcpy(h_colors.data(), colors, sizeof(int) * (vertices + 1), cudaMemcpyDeviceToHost);
    int total_colors = 0;
    for (int i = 0; i <= vertices; ++i) {
        total_colors += (h_colors[i] != 0);
    }
    printf("Total colors used: %d\n", total_colors);
    printf("[INFO] Done. Cleaning up.\n");
    free(h_vertex_pointers);
    free(h_adjacencies);
    cudaFree(uncolored_count);
    cudaFree(conflict_count);
    cudaFree(colors);
    cudaFree(n_colored);
    cudaFree(vertex_pointers);
    cudaFree(adjacencies);
    cudaFree(coloring);
    cudaFree(candidates);
    
    // Conditionally free sorting-related allocations
    if (d_priorities) cudaFree(d_priorities);
    if (d_degrees) cudaFree(d_degrees);
    if (d_n_marked) cudaFree(d_n_marked);
    if (sorted_coloring) cudaFree(sorted_coloring);
    if (sorted_candidates) cudaFree(sorted_candidates);
    if (useSorting) free_sorted_graph(sorted_graph);
    if (d_random_prio) cudaFree(d_random_prio);

    return 0;
}