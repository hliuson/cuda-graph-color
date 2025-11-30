#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

static void load_edges_to_csc(const char *path,
                              std::vector<int> &colPtr,
                              std::vector<int> &rowIdx,
                              int &vertices,
                              int &edges,
                              bool zeroBaseInput = true) {
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
        std::istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue;
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
    
    // Phase 1: Mark candidates (vertices with no higher-ID uncolored neighbor)
    for (int i = tid; i < vertices; i += gridsize) {
        candidates[i] = 0;
        if (coloring[i] != -1) continue;  // Already colored
        
        int is_candidate = 1;
        for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
            int neighbor = adjacencies[j];
            if (neighbor > i && coloring[neighbor] == -1) {
                is_candidate = 0;
                break;
            }
        }
        candidates[i] = is_candidate;
    }
}

__global__ void apply_coloring(int *vertex_pointers, int *adjacencies, int vertices,
                               int *coloring, int *candidates, int *n_colored, int iteration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridsize = blockDim.x * gridDim.x;
    
    int colored = 0;
    for (int i = tid; i < vertices; i += gridsize) {
        if (candidates[i] && coloring[i] == -1) {
            // Check no lower-ID neighbor is also a candidate
            int safe = 1;
            for (int j = vertex_pointers[i]; j < vertex_pointers[i + 1]; j++) {
                int neighbor = adjacencies[j];
                if (neighbor < i && candidates[neighbor]) {
                    safe = 0;
                    break;
                }
            }
            if (safe) {
                coloring[i] = iteration;
                colored++;
            }
        }
    }
    
    // Reduction code for colored count...
    __syncthreads();
    colored += __shfl_down_sync(0xFFFFFFFF, colored, 16);
    colored += __shfl_down_sync(0xFFFFFFFF, colored, 8);
    colored += __shfl_down_sync(0xFFFFFFFF, colored, 4);
    colored += __shfl_down_sync(0xFFFFFFFF, colored, 2);
    colored += __shfl_down_sync(0xFFFFFFFF, colored, 1);
    
    __shared__ int warp_sum[32];
    if (threadIdx.x % 32 == 0) {
        warp_sum[threadIdx.x / 32] = colored;
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        colored = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sum[threadIdx.x] : 0;
        colored += __shfl_down_sync(0xFFFFFFFF, colored, 16);
        colored += __shfl_down_sync(0xFFFFFFFF, colored, 8);
        colored += __shfl_down_sync(0xFFFFFFFF, colored, 4);
        colored += __shfl_down_sync(0xFFFFFFFF, colored, 2);
        colored += __shfl_down_sync(0xFFFFFFFF, colored, 1);
    }
    if (threadIdx.x == 0) {
        n_colored[blockIdx.x] = colored;
    }
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

int main(int argc, char *argv[]) {
    int *h_adjacencies = nullptr;
    int *h_vertex_pointers = nullptr;
    int edges = 0;
    int vertices = 0;
    // Build CSC from edge list
    const char *edgeFile = (argc > 1) ? argv[1] : "1684.edges";
    std::vector<int> colPtr, rowIdx;
    // Assume input is 0-based; set false if file is 1-based
    printf("[INFO] Parsing edges from: %s\n", edgeFile);
    load_edges_to_csc(edgeFile, colPtr, rowIdx, vertices, edges, true);
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
        luby_mis_coloring_iter<<<numBlocks, blockSize>>>(vertex_pointers, adjacencies, vertices, coloring, candidates, n_colored, iter);
        apply_coloring<<<numBlocks, blockSize>>>(vertex_pointers, adjacencies, vertices, coloring, candidates, n_colored, iter);
        cudaDeviceSynchronize();
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