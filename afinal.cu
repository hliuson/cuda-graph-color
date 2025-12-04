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
// Partition-based greedy coloring
// Colors one priority partition at a time
// ============================================================================

// Color a single partition using greedy algorithm
// Vertices in [start_idx, end_idx) are colored
// coloring: array of size total_vertices (in sorted order)
// All neighbors (including cross-partition) are considered for conflicts
__global__ void color_partition_greedy(
    int *vertex_pointers, int *adjacencies,
    int start_idx, int end_idx,
    int *coloring) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    const int MAX_COLOR_WORDS = 8;  // Supports up to 256 colors
    
    for (int i = start_idx + tid; i < end_idx; i += gridsize) {
        // Skip if already colored
        if (coloring[i] != -1) continue;
        
        // Build used color mask from ALL neighbors
        unsigned int used_colors[MAX_COLOR_WORDS] = {0};
        
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            int nc = coloring[neighbor];
            if (nc >= 0 && nc < MAX_COLOR_WORDS * 32) {
                used_colors[nc / 32] |= (1u << (nc % 32));
            }
        }
        
        // TODO: Stub - actual color assignment goes here
        // For now, find first available color (greedy)
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

// Host function to color all partitions in priority order
// Colors are assigned in the sorted graph's vertex order
// Returns coloring array (in sorted order) - use perm to map back to original
void color_by_partition(
    SortedGraph &sg,
    int *d_coloring,          // output: size vertices, caller allocates
    int *h_priority_offsets,  // host copy of priority offsets
    int numBlocks, int blockSize) {
    
    // Initialize all vertices as uncolored
    cudaMemset(d_coloring, -1, sizeof(int) * sg.vertices);
    
    // Color partitions in reverse priority order (highest priority = colored last = lowest degree)
    // This way, when we color partition p, all higher-priority partitions are already colored
    for (int p = sg.num_priorities - 1; p >= 0; p--) {
        int start_idx = h_priority_offsets[p];
        int end_idx = h_priority_offsets[p + 1];
        
        if (start_idx >= end_idx) continue;  // Empty partition
        
        printf("[INFO] Coloring partition %d: vertices [%d, %d)\n", p, start_idx, end_idx);
        
        color_partition_greedy<<<numBlocks, blockSize>>>(
            sg.vertex_pointers, sg.adjacencies,
            start_idx, end_idx,
            d_coloring);
        
        cudaDeviceSynchronize();
    }
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

__global__ void blockwise_mis_color(int *vertex_pointers, int *adjacencies, int vertices, 
                        int *coloring, int *candidates, int *n_colored) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int colored = 0;
    int done = 0;
    while (!done) {

    }
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
    //to get a list of vertices adjacent to vertex i, use adjncy[xadj[i]] to adjncy[xadj[i+1]-1]
    int *adjacencies, *vertex_pointers, *coloring;
    int *n_colored;
    int *candidates;

    
    int blockSize = 256;
    int numBlocks = 16; //we use grid-stride loops everywhere, so this is arbitrary

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


    cudaMemset(coloring, -1, sizeof(int) * vertices);
    int cum_colored = 0;
    printf("[INFO] Starting Luby MIS coloring iterations...\n");
    int iter = 0;

    while (cum_colored < vertices) {
            //time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        luby_mis_coloring_iter<<<numBlocks, blockSize>>>(vertex_pointers, adjacencies, vertices, coloring, candidates, n_colored, iter);
        apply_coloring<<<numBlocks, blockSize>>>(vertex_pointers, adjacencies, vertices, coloring, candidates, n_colored, iter);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("[INFO] Iteration %d coloring time: %f ms\n", iter, milliseconds);

        int h_n_colored[numBlocks];
        int new_colored = 0;
        cudaMemcpy(h_n_colored, n_colored, sizeof(int) * numBlocks, cudaMemcpyDeviceToHost);
        for (int i = 0; i < numBlocks; i++) { //have to copy to CPU to do control flow with kernel launches 
            new_colored += h_n_colored[i];
        }
        cum_colored += new_colored;
        printf("[INFO] Iteration %d colored=%d, cumulative colored=%d/%d\n", iter, new_colored, cum_colored, vertices);
        iter++;
    }
    printf("[INFO] Coloring complete.\n");
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

    return 0;
}