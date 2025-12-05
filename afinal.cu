#include <cstdio>
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

// ============================================================================
// Partition-based greedy coloring with binary search on color space
// Uses O(1) space per thread, O(log(max_color)) iterations
// ============================================================================

// Fixed-size bitmask for tracking used colors in a range [lo, lo+MASK_SIZE)
#define COLOR_MASK_SIZE 64

// Check which colors in [color_lo, color_lo + COLOR_MASK_SIZE) are used by neighbors
// Returns a bitmask where bit i is set if color (color_lo + i) is used
__device__ unsigned long long get_used_colors_in_range(
    int *vertex_pointers, int *adjacencies,
    int vertex, int *coloring,
    int color_lo) {
    
    unsigned long long used = 0ULL;
    int color_hi = color_lo + COLOR_MASK_SIZE;
    
    for (int j = vertex_pointers[vertex]; j < vertex_pointers[vertex + 1]; j++) {
        int neighbor = adjacencies[j];
        int nc = coloring[neighbor];
        if (nc >= color_lo && nc < color_hi) {
            used |= (1ULL << (nc - color_lo));
        }
    }
    return used;
}

// Find first available color using binary search + fixed-size window
// Returns the smallest color not used by any neighbor
__device__ int find_first_available_color(
    int *vertex_pointers, int *adjacencies,
    int vertex, int *coloring,
    int max_color_hint) {
    
    // Binary search: find the first window [lo, lo+64) that has an available color
    // Start with lo=0, double until we find availability, then binary search down
    
    int lo = 0;
    
    // Phase 1: Scan windows of size COLOR_MASK_SIZE until we find one with available color
    while (lo < max_color_hint) {
        unsigned long long used = get_used_colors_in_range(
            vertex_pointers, adjacencies, vertex, coloring, lo);
        
        if (~used != 0ULL) {
            // Found a window with at least one available color
            // Return the first available in this window
            return lo + __ffsll(~used) - 1;
        }
        lo += COLOR_MASK_SIZE;
    }
    
    // All colors in [0, max_color_hint) are used, check beyond
    // (This should be rare if max_color_hint is set well)
    unsigned long long used = get_used_colors_in_range(
        vertex_pointers, adjacencies, vertex, coloring, lo);
    return lo + __ffsll(~used) - 1;
}

// Color a single partition using greedy algorithm with binary search
// Vertices in [start_idx, end_idx) are colored
// coloring: array of size total_vertices (in sorted order)
// All neighbors (including cross-partition) are considered for conflicts
__global__ void color_partition_greedy(
    int *vertex_pointers, int *adjacencies,
    int start_idx, int end_idx,
    int *coloring,
    int max_color_hint) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int i = start_idx + tid; i < end_idx; i += gridsize) {
        // Skip if already colored
        if (coloring[i] != -1) continue;
        
        int color = find_first_available_color(
            vertex_pointers, adjacencies, i, coloring, max_color_hint);
        
        coloring[i] = color;
    }
}

// ============================================================================
// Deconfliction: resolve color conflicts by having lower-degree vertex recolor
// ============================================================================

// Detect conflicts and mark lower-degree vertices for recoloring
// Returns number of conflicts found (via reduction)
__global__ void detect_conflicts_mark_lower_degree(
    int *vertex_pointers, int *adjacencies, int vertices,
    int *coloring, int *needs_recolor, int *conflict_count) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int local_conflicts = 0;
    
    for (int i = tid; i < vertices; i += gridsize) {
        int my_color = coloring[i];
        if (my_color == -1) continue;
        
        int my_degree = vertex_pointers[i + 1] - vertex_pointers[i];
        
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            int neighbor_color = coloring[neighbor];
            
            if (my_color == neighbor_color && neighbor > i) {
                // Found a conflict - only count once (when i < neighbor)
                local_conflicts++;
                
                int neighbor_degree = vertex_pointers[neighbor + 1] - vertex_pointers[neighbor];
                
                // Lower degree vertex recolors; tie-break by vertex ID
                if (my_degree < neighbor_degree || (my_degree == neighbor_degree && i < neighbor)) {
                    needs_recolor[i] = 1;
                } else {
                    needs_recolor[neighbor] = 1;
                }
            }
        }
    }
    
    __syncthreads();
    reduce(local_conflicts, conflict_count);
}

// Recolor vertices that were marked for recoloring
__global__ void recolor_marked_vertices(
    int *vertex_pointers, int *adjacencies, int vertices,
    int *coloring, int *needs_recolor, int max_color_hint) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    for (int i = tid; i < vertices; i += gridsize) {
        if (needs_recolor[i] == 1) {
            // Clear current color first so we can find a new one
            coloring[i] = -1;
        }
    }
    
    __syncthreads();
    
    for (int i = tid; i < vertices; i += gridsize) {
        if (needs_recolor[i] == 1) {
            int color = find_first_available_color(
                vertex_pointers, adjacencies, i, coloring, max_color_hint);
            coloring[i] = color;
        }
    }
}

// Host function to color all partitions in priority order
// Colors are assigned in the sorted graph's vertex order
// Returns coloring array (in sorted order) - use perm to map back to original
void color_by_partition(
    SortedGraph &sg,
    int *d_coloring,          // output: size vertices, caller allocates
    int *h_priority_offsets,  // host copy of priority offsets
    int max_color_hint,       // hint for max colors (e.g., max_degree + 1)
    int numBlocks, int blockSize) {
    
    // Initialize all vertices as uncolored
    cudaMemset(d_coloring, -1, sizeof(int) * sg.vertices);
    
    // Allocate deconfliction buffers once
    int *d_needs_recolor, *d_conflict_count;
    cudaMalloc(&d_needs_recolor, sizeof(int) * sg.vertices);
    cudaMalloc(&d_conflict_count, sizeof(int) * numBlocks);
    
    // Pre-compute average degree for each partition (outside of timing)
    // Copy vertex pointers to host for degree computation
    std::vector<int> h_vertex_pointers(sg.vertices + 1);
    cudaMemcpy(h_vertex_pointers.data(), sg.vertex_pointers, 
               sizeof(int) * (sg.vertices + 1), cudaMemcpyDeviceToHost);
    
    std::vector<float> partition_avg_degree(sg.num_priorities, 0.0f);
    for (int p = 0; p < sg.num_priorities; p++) {
        int start_idx = h_priority_offsets[p];
        int end_idx = h_priority_offsets[p + 1];
        if (start_idx >= end_idx) continue;
        
        int partition_size = end_idx - start_idx;
        // Sum degrees: sum of (vp[i+1] - vp[i]) for i in [start_idx, end_idx)
        // = vp[end_idx] - vp[start_idx]
        int total_edges = h_vertex_pointers[end_idx] - h_vertex_pointers[start_idx];
        partition_avg_degree[p] = (float)total_edges / partition_size;
    }
    
    // Timing events for per-partition timing
    cudaEvent_t part_start, part_stop;
    cudaEventCreate(&part_start);
    cudaEventCreate(&part_stop);
    
    // Color partitions in reverse priority order (highest priority = colored last = lowest degree)
    // This way, when we color partition p, all higher-priority partitions are already colored
    for (int p = sg.num_priorities - 1; p >= 0; p--) {
        int start_idx = h_priority_offsets[p];
        int end_idx = h_priority_offsets[p + 1];
        
        if (start_idx >= end_idx) continue;  // Empty partition
        
        int partition_size = end_idx - start_idx;
        float avg_degree = partition_avg_degree[p];
        
        cudaEventRecord(part_start);
        
        color_partition_greedy<<<numBlocks, blockSize>>>(
            sg.vertex_pointers, sg.adjacencies,
            start_idx, end_idx,
            d_coloring,
            max_color_hint);
        
        cudaDeviceSynchronize();
        
        // ====================================================================
        // Deconfliction phase after each partition: resolve conflicts by 
        // having lower-degree vertex recolor
        // ====================================================================
        int deconflict_iterations = 0;
        int max_deconflict_iterations = 100;  // Safety limit
        
        while (deconflict_iterations < max_deconflict_iterations) {
            // Reset markers
            cudaMemset(d_needs_recolor, 0, sizeof(int) * sg.vertices);
            cudaMemset(d_conflict_count, 0, sizeof(int) * numBlocks);
            
            // Detect conflicts and mark lower-degree vertices
            detect_conflicts_mark_lower_degree<<<numBlocks, blockSize>>>(
                sg.vertex_pointers, sg.adjacencies, sg.vertices,
                d_coloring, d_needs_recolor, d_conflict_count);
            cudaDeviceSynchronize();
            
            // Sum up conflicts from all blocks
            int h_conflict_count[numBlocks];
            cudaMemcpy(h_conflict_count, d_conflict_count, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
            int total_conflicts = 0;
            for (int i = 0; i < numBlocks; i++) {
                total_conflicts += h_conflict_count[i];
            }
            
            if (total_conflicts == 0) {
                break;
            }
            
            // Recolor marked vertices
            recolor_marked_vertices<<<numBlocks, blockSize>>>(
                sg.vertex_pointers, sg.adjacencies, sg.vertices,
                d_coloring, d_needs_recolor, max_color_hint);
            cudaDeviceSynchronize();
            
            deconflict_iterations++;
        }
        
        cudaEventRecord(part_stop);
        cudaEventSynchronize(part_stop);
        float part_ms;
        cudaEventElapsedTime(&part_ms, part_start, part_stop);
        float time_per_node_us = (part_ms * 1000.0f) / partition_size;
        
        printf("[INFO] Partition %d: %d vertices, avg_deg=%.1f, %.2f ms (%.3f us/node), %d deconflict iters\n", 
               p, partition_size, avg_degree, part_ms, time_per_node_us, deconflict_iterations);
        
        if (deconflict_iterations >= max_deconflict_iterations) {
            printf("[WARN] Partition %d deconfliction reached max iterations (%d)\n", p, max_deconflict_iterations);
        }
    }
    
    cudaEventDestroy(part_start);
    cudaEventDestroy(part_stop);
    cudaFree(d_needs_recolor);
    cudaFree(d_conflict_count);
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
    // Build CSC from edge list
    const char *edgeFile = (argc > 1) ? argv[1] : "1684.edges";
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
    
    int blockSize = 256;
    int numBlocks = 640;
    
    // ========================================================================
    // Step 1: Copy graph to device
    // ========================================================================
    int *d_vertex_pointers, *d_adjacencies;
    printf("[INFO] Allocating device memory...\n");
    cudaMalloc(&d_vertex_pointers, sizeof(int) * (vertices + 1));
    cudaMalloc(&d_adjacencies, sizeof(int) * edges);
    cudaMemcpy(d_vertex_pointers, h_vertex_pointers, sizeof(int) * (vertices + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacencies, h_adjacencies, sizeof(int) * edges, cudaMemcpyHostToDevice);
    printf("[INFO] Host-to-device copy complete.\n");
    
    // ========================================================================
    // Step 2: Compute smallest-last ordering (priorities)
    // ========================================================================
    int *d_priorities, *d_degrees, *d_n_marked;
    cudaMalloc(&d_priorities, sizeof(int) * vertices);
    cudaMalloc(&d_degrees, sizeof(int) * vertices);
    cudaMalloc(&d_n_marked, sizeof(int) * numBlocks);
    
    // Initialize priorities to -1 (active), degrees to actual degree
    cudaMemset(d_priorities, -1, sizeof(int) * vertices);
    
    // Compute initial degrees
    thrust::device_ptr<int> vp_ptr(d_vertex_pointers);
    thrust::device_ptr<int> deg_ptr(d_degrees);
    thrust::transform(vp_ptr + 1, vp_ptr + 1 + vertices, vp_ptr, deg_ptr, thrust::minus<int>());
    
    // Compute max degree BEFORE smallest-last ordering modifies d_degrees
    int max_degree = thrust::reduce(deg_ptr, deg_ptr + vertices, 0, thrust::maximum<int>());
    int max_color_hint = max_degree + 1;
    printf("[INFO] Max degree: %d, color hint: %d\n", max_degree, max_color_hint);
    
    printf("[INFO] Computing smallest-last ordering...\n");
    cudaEvent_t slo_start, slo_stop;
    cudaEventCreate(&slo_start);
    cudaEventCreate(&slo_stop);
    cudaEventRecord(slo_start);
    
    int num_priorities = 0;
    int remaining = vertices;
    float eps = 0.1f;
    
    while (remaining > 0) {
        float avg_degree = get_average_degree(d_vertex_pointers, d_adjacencies, vertices, d_priorities);
        
        cudaMemset(d_n_marked, 0, sizeof(int) * numBlocks);
        smallest_last_ordering<<<numBlocks, blockSize>>>(
            d_vertex_pointers, d_adjacencies, vertices, remaining,
            num_priorities, d_priorities, d_degrees, (int)avg_degree, eps, d_n_marked);
        
        // Count marked vertices
        int h_n_marked[numBlocks];
        cudaMemcpy(h_n_marked, d_n_marked, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
        int marked = 0;
        for (int i = 0; i < numBlocks; i++) marked += h_n_marked[i];
        
        if (marked == 0) {
            // No vertices marked, increase threshold
            eps *= 2.0f;
            continue;
        }
        
        // Update degrees for remaining active vertices
        update_degrees<<<numBlocks, blockSize>>>(
            d_vertex_pointers, d_adjacencies, vertices, num_priorities, d_priorities, d_degrees);
        
        remaining -= marked;
        num_priorities++;
        eps = 0.1f;  // Reset epsilon
        
        printf("[INFO] Priority %d: marked %d vertices, remaining %d\n", num_priorities - 1, marked, remaining);
    }
    
    cudaEventRecord(slo_stop);
    cudaEventSynchronize(slo_stop);
    float slo_ms;
    cudaEventElapsedTime(&slo_ms, slo_start, slo_stop);
    printf("[INFO] Smallest-last ordering complete: %d priority levels, %.2f ms\n", num_priorities, slo_ms);
    
    // ========================================================================
    // Step 3: Sort graph by priority
    // ========================================================================
    printf("[INFO] Sorting graph by priority...\n");
    cudaEvent_t sort_start, sort_stop;
    cudaEventCreate(&sort_start);
    cudaEventCreate(&sort_stop);
    cudaEventRecord(sort_start);
    
    SortedGraph sg;
    sort_graph_by_priority(d_vertex_pointers, d_adjacencies, vertices, edges,
                           d_priorities, num_priorities, sg, numBlocks, blockSize);
    
    cudaEventRecord(sort_stop);
    cudaEventSynchronize(sort_stop);
    float sort_ms;
    cudaEventElapsedTime(&sort_ms, sort_start, sort_stop);
    printf("[INFO] Graph sorted: %d vertices, %d edges, %.2f ms\n", sg.vertices, sg.edges, sort_ms);
    
    // Copy priority offsets to host
    int *h_priority_offsets = (int*)malloc(sizeof(int) * (num_priorities + 1));
    cudaMemcpy(h_priority_offsets, sg.priority_offsets, sizeof(int) * (num_priorities + 1), cudaMemcpyDeviceToHost);
    
    // ========================================================================
    // Step 4: Color partitions
    // ========================================================================
    printf("[INFO] Coloring partitions...\n");
    cudaEvent_t color_start, color_stop;
    cudaEventCreate(&color_start);
    cudaEventCreate(&color_stop);
    cudaEventRecord(color_start);
    
    int *d_sorted_coloring;
    cudaMalloc(&d_sorted_coloring, sizeof(int) * vertices);
    
    color_by_partition(sg, d_sorted_coloring, h_priority_offsets, max_color_hint, numBlocks, blockSize);
    
    cudaEventRecord(color_stop);
    cudaEventSynchronize(color_stop);
    float color_ms;
    cudaEventElapsedTime(&color_ms, color_start, color_stop);
    printf("[INFO] Coloring complete: %.2f ms\n", color_ms);
    
    // ========================================================================
    // Step 5: Unmap coloring back to original order
    // ========================================================================
    int *d_coloring;
    cudaMalloc(&d_coloring, sizeof(int) * vertices);
    unmap_coloring<<<numBlocks, blockSize>>>(d_sorted_coloring, d_coloring, sg.perm, vertices);
    cudaDeviceSynchronize();
    
    // ========================================================================
    // Step 6: Verify coloring
    // ========================================================================
    printf("[INFO] Verifying coloring...\n");
    int *d_colors, *d_uncolored_count, *d_conflict_count;
    cudaMalloc(&d_colors, sizeof(int) * (vertices + 1));
    cudaMalloc(&d_uncolored_count, sizeof(int));
    cudaMalloc(&d_conflict_count, sizeof(int));
    cudaMemset(d_colors, 0, sizeof(int) * (vertices + 1));
    cudaMemset(d_uncolored_count, 0, sizeof(int));
    cudaMemset(d_conflict_count, 0, sizeof(int));
    
    verify_coloring<<<numBlocks, blockSize>>>(
        d_vertex_pointers, d_adjacencies, vertices, d_coloring,
        d_colors, d_uncolored_count, d_conflict_count);
    cudaDeviceSynchronize();
    
    int h_uncolored_count, h_conflict_count;
    cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_conflict_count, d_conflict_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Count colors used
    std::vector<int> h_colors(vertices + 1);
    cudaMemcpy(h_colors.data(), d_colors, sizeof(int) * (vertices + 1), cudaMemcpyDeviceToHost);
    int total_colors = 0;
    for (int i = 0; i <= vertices; ++i) {
        total_colors += (h_colors[i] != 0);
    }
    
    printf("\n========== PARTITION-BASED RESULTS ==========\n");
    printf("Total vertices: %d\n", vertices);
    printf("Total edges: %d\n", edges);
    printf("Priority levels: %d\n", num_priorities);
    printf("Uncolored vertices: %d\n", h_uncolored_count);
    printf("Conflicting vertices: %d\n", h_conflict_count);
    printf("Total colors used: %d\n", total_colors);
    printf("Ordering time: %.2f ms\n", slo_ms);
    printf("Sort time: %.2f ms\n", sort_ms);
    printf("Coloring time: %.2f ms\n", color_ms);
    printf("Total time: %.2f ms\n", slo_ms + sort_ms + color_ms);
    printf("=============================================\n");
    
    // ========================================================================
    // Step 7: Run Luby MIS coloring for comparison
    // ========================================================================
    printf("\n[INFO] Running Luby MIS coloring for comparison...\n");
    
    int *d_luby_coloring, *d_candidates, *d_n_colored;
    cudaMalloc(&d_luby_coloring, sizeof(int) * vertices);
    cudaMalloc(&d_candidates, sizeof(int) * vertices);
    cudaMalloc(&d_n_colored, sizeof(int) * numBlocks);
    
    // Initialize coloring to -1 (uncolored)
    cudaMemset(d_luby_coloring, -1, sizeof(int) * vertices);
    
    cudaEvent_t luby_start, luby_stop;
    cudaEventCreate(&luby_start);
    cudaEventCreate(&luby_stop);
    cudaEventRecord(luby_start);
    
    int luby_iteration = 0;
    int luby_remaining = vertices;
    int max_luby_iterations = 1000;  // Safety limit
    
    while (luby_remaining > 0 && luby_iteration < max_luby_iterations) {
        // Phase 1: Speculatively color vertices
        luby_mis_coloring_iter<<<numBlocks, blockSize>>>(
            d_vertex_pointers, d_adjacencies, vertices,
            d_luby_coloring, d_candidates, d_n_colored, luby_iteration);
        cudaDeviceSynchronize();
        
        // Phase 2: Apply coloring (resolve conflicts by vertex ID)
        cudaMemset(d_n_colored, 0, sizeof(int) * numBlocks);
        apply_coloring<<<numBlocks, blockSize>>>(
            d_vertex_pointers, d_adjacencies, vertices,
            d_luby_coloring, d_candidates, d_n_colored, luby_iteration);
        cudaDeviceSynchronize();
        
        // Count newly colored vertices
        int h_n_colored[numBlocks];
        cudaMemcpy(h_n_colored, d_n_colored, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
        int colored_this_iter = 0;
        for (int i = 0; i < numBlocks; i++) colored_this_iter += h_n_colored[i];
        
        luby_remaining -= colored_this_iter;
        luby_iteration++;
        
        if (luby_iteration % 10 == 0 || colored_this_iter == 0) {
            printf("[INFO] Luby iteration %d: colored %d, remaining %d\n", 
                   luby_iteration, colored_this_iter, luby_remaining);
        }
        
        if (colored_this_iter == 0 && luby_remaining > 0) {
            // No progress made but vertices remain - shouldn't happen with correct algorithm
            printf("[WARN] Luby made no progress with %d vertices remaining\n", luby_remaining);
            break;
        }
    }
    
    cudaEventRecord(luby_stop);
    cudaEventSynchronize(luby_stop);
    float luby_ms;
    cudaEventElapsedTime(&luby_ms, luby_start, luby_stop);
    
    // Verify Luby coloring
    int *d_luby_colors, *d_luby_uncolored, *d_luby_conflicts;
    cudaMalloc(&d_luby_colors, sizeof(int) * (vertices + 1));
    cudaMalloc(&d_luby_uncolored, sizeof(int));
    cudaMalloc(&d_luby_conflicts, sizeof(int));
    cudaMemset(d_luby_colors, 0, sizeof(int) * (vertices + 1));
    cudaMemset(d_luby_uncolored, 0, sizeof(int));
    cudaMemset(d_luby_conflicts, 0, sizeof(int));
    
    verify_coloring<<<numBlocks, blockSize>>>(
        d_vertex_pointers, d_adjacencies, vertices, d_luby_coloring,
        d_luby_colors, d_luby_uncolored, d_luby_conflicts);
    cudaDeviceSynchronize();
    
    int h_luby_uncolored, h_luby_conflicts;
    cudaMemcpy(&h_luby_uncolored, d_luby_uncolored, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_luby_conflicts, d_luby_conflicts, sizeof(int), cudaMemcpyDeviceToHost);
    
    std::vector<int> h_luby_colors(vertices + 1);
    cudaMemcpy(h_luby_colors.data(), d_luby_colors, sizeof(int) * (vertices + 1), cudaMemcpyDeviceToHost);
    int luby_total_colors = 0;
    for (int i = 0; i <= vertices; ++i) {
        luby_total_colors += (h_luby_colors[i] != 0);
    }
    
    printf("\n========== LUBY MIS RESULTS ==========\n");
    printf("Total vertices: %d\n", vertices);
    printf("Total edges: %d\n", edges);
    printf("Iterations: %d\n", luby_iteration);
    printf("Uncolored vertices: %d\n", h_luby_uncolored);
    printf("Conflicting vertices: %d\n", h_luby_conflicts);
    printf("Total colors used: %d\n", luby_total_colors);
    printf("Coloring time: %.2f ms\n", luby_ms);
    printf("=======================================\n");
    
    printf("\n========== COMPARISON ==========\n");
    printf("                    Partition    Luby MIS\n");
    printf("Colors used:        %-12d %d\n", total_colors, luby_total_colors);
    printf("Conflicts:          %-12d %d\n", h_conflict_count, h_luby_conflicts);
    printf("Time (ms):          %-12.2f %.2f\n", slo_ms + sort_ms + color_ms, luby_ms);
    printf("=================================\n");
    
    // Cleanup Luby arrays
    cudaFree(d_luby_coloring);
    cudaFree(d_candidates);
    cudaFree(d_n_colored);
    cudaFree(d_luby_colors);
    cudaFree(d_luby_uncolored);
    cudaFree(d_luby_conflicts);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("[INFO] Cleaning up.\n");
    free(h_vertex_pointers);
    free(h_adjacencies);
    free(h_priority_offsets);
    
    cudaFree(d_vertex_pointers);
    cudaFree(d_adjacencies);
    cudaFree(d_priorities);
    cudaFree(d_degrees);
    cudaFree(d_n_marked);
    cudaFree(d_sorted_coloring);
    cudaFree(d_coloring);
    cudaFree(d_colors);
    cudaFree(d_uncolored_count);
    cudaFree(d_conflict_count);
    
    free_sorted_graph(sg);
    
    return 0;
}