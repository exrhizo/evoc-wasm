//! Port of the heap/candidate machinery of `evoc/common_nndescent.py`
//! (`make_heap`, `siftdown`, `deheap_sort`, `build_candidates_heap_push`,
//! `build_candidates`, `flagged_heap_push`, `apply_graph_update_array`,
//! `apply_sorted_graph_updates`). `seed` / `tau_rand_int` / `tau_rand`
//! live in `rng.rs`.
//!
//! PORT NOTE: every `numba.prange` here becomes a plain sequential loop
//! (single-threaded wasm32 target), but the *block structure* driven by
//! `n_threads` is kept intact: block boundaries determine which heap rows a
//! "thread" pass touches and therefore the exact push order and heap
//! contents. Fixtures are generated with n_threads=1.

use crate::rng::{tau_rand, RngState};

/// Port of the `(indices, distances, flags)` heap tuple returned by
/// `make_heap`. Row-major n_points x size.
pub struct NNDHeap {
    pub n_points: usize,
    pub size: usize,
    pub indices: Vec<i32>,
    pub distances: Vec<f32>,
    pub flags: Vec<u8>,
}

impl NNDHeap {
    /// Flat range of row `i` (shared by all three arrays).
    pub fn row(&self, i: usize) -> std::ops::Range<usize> {
        i * self.size..(i + 1) * self.size
    }
}

/// Port of `make_heap`: indices -1, distances +inf, flags 0.
pub fn make_heap(n_points: usize, size: usize) -> NNDHeap {
    NNDHeap {
        n_points,
        size,
        indices: vec![-1; n_points * size],
        distances: vec![f32::INFINITY; n_points * size],
        flags: vec![0; n_points * size],
    }
}

/// Port of common_nndescent's `siftdown`: restore the max-heap property for
/// a heap with an out of place element at position `elt`. `heap1` carries
/// the weights and `heap2` the corresponding elements.
pub fn siftdown(heap1: &mut [f32], heap2: &mut [i32], elt: usize) {
    let mut elt = elt;
    while elt * 2 + 1 < heap1.len() {
        let left_child = elt * 2 + 1;
        let right_child = left_child + 1;
        let mut swap = elt;

        if heap1[swap] < heap1[left_child] {
            swap = left_child;
        }

        if right_child < heap1.len() && heap1[swap] < heap1[right_child] {
            swap = right_child;
        }

        if swap == elt {
            break;
        } else {
            heap1.swap(elt, swap);
            heap2.swap(elt, swap);
            elt = swap;
        }
    }
}

/// Port of common_nndescent's `deheap_sort`: reorder each row of the paired
/// (indices, distances) heap matrices by increasing distance, in place.
///
/// PORT NOTE: the argument order `(indices, distances)` is the OPPOSITE of
/// `kdtree::deheap_sort(distances, indices)`. Each module keeps its own
/// reference file's convention -- do not "harmonize" them. (The reference's
/// `prange` over rows is sequential here; rows are independent.)
pub fn deheap_sort(indices: &mut [i32], distances: &mut [f32], n_cols: usize) {
    let n_rows = indices.len() / n_cols;
    for i in 0..n_rows {
        let row_i = &mut indices[i * n_cols..(i + 1) * n_cols];
        let row_d = &mut distances[i * n_cols..(i + 1) * n_cols];
        // starting from the end of the array and moving back
        for j in (1..n_cols).rev() {
            row_i.swap(0, j);
            row_d.swap(0, j);

            siftdown(&mut row_d[..j], &mut row_i[..j], 0);
        }
    }
}

/// Port of `build_candidates_heap_push`: max-heap push that also rejects
/// elements already present.
///
/// PORT NOTE: reference numba locals use u16 loop indices -- behaviorally
/// irrelevant for heap sizes < 32768; we use usize.
pub fn build_candidates_heap_push(
    priorities: &mut [f32],
    indices: &mut [i32],
    p: f32,
    n: i32,
) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    // break if we already have this element.
    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    // insert val at position zero
    priorities[0] = p;
    indices[0] = n;

    // descend the heap, swapping values until the max heap criterion is met
    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            i_swap = ic2;
        } else {
            break;
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];

        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;

    1
}

/// Port of `build_candidates`: build heaps of new/old candidate neighbors
/// for NN-descent. Returns `(new_candidate_indices, old_candidate_indices)`
/// as flat row-major n_vertices x max_candidates arrays. Mutates the
/// graph's flags (marking retained new-candidates as no longer "new").
///
/// PORT NOTE: the reference's `local_rng_state = rng_state + n` is a numpy
/// array-plus-scalar, which allocates a NEW array (each lane + n). All
/// tau_rand draws advance that per-thread copy, so the caller's rng_state
/// is NEVER advanced by this function (the fixture's `rng_state_after`
/// equals the input, confirming). Hence `rng_state: &RngState` (shared).
pub fn build_candidates(
    current_graph: &mut NNDHeap,
    max_candidates: usize,
    rng_state: &RngState,
    n_threads: usize,
) -> (Vec<i32>, Vec<i32>) {
    let n_vertices = current_graph.n_points;
    let n_neighbors = current_graph.size;

    let mut new_candidate_indices = vec![-1i32; n_vertices * max_candidates];
    let mut new_candidate_priority = vec![f32::INFINITY; n_vertices * max_candidates];

    let mut old_candidate_indices = vec![-1i32; n_vertices * max_candidates];
    let mut old_candidate_priority = vec![f32::INFINITY; n_vertices * max_candidates];

    let block_size = n_vertices / n_threads + 1;

    for n in 0..n_threads {
        let mut local_rng_state: RngState = [
            rng_state[0].wrapping_add(n as i64),
            rng_state[1].wrapping_add(n as i64),
            rng_state[2].wrapping_add(n as i64),
        ];
        let block_start = n * block_size;
        let block_end = std::cmp::min(block_start + block_size, n_vertices);

        for i in 0..n_vertices {
            for j in 0..n_neighbors {
                let idx = current_graph.indices[i * n_neighbors + j] as i64;

                if idx >= 0
                    && ((i >= block_start && i < block_end)
                        || (idx >= block_start as i64 && idx < block_end as i64))
                {
                    let isn = current_graph.flags[i * n_neighbors + j];
                    let d = tau_rand(&mut local_rng_state);

                    if isn != 0 {
                        if i >= block_start && i < block_end {
                            build_candidates_heap_push(
                                &mut new_candidate_priority
                                    [i * max_candidates..(i + 1) * max_candidates],
                                &mut new_candidate_indices
                                    [i * max_candidates..(i + 1) * max_candidates],
                                d,
                                idx as i32,
                            );
                        }
                        if idx >= block_start as i64 && idx < block_end as i64 {
                            let idx = idx as usize;
                            build_candidates_heap_push(
                                &mut new_candidate_priority
                                    [idx * max_candidates..(idx + 1) * max_candidates],
                                &mut new_candidate_indices
                                    [idx * max_candidates..(idx + 1) * max_candidates],
                                d,
                                i as i32,
                            );
                        }
                    } else {
                        if i >= block_start && i < block_end {
                            build_candidates_heap_push(
                                &mut old_candidate_priority
                                    [i * max_candidates..(i + 1) * max_candidates],
                                &mut old_candidate_indices
                                    [i * max_candidates..(i + 1) * max_candidates],
                                d,
                                idx as i32,
                            );
                        }
                        if idx >= block_start as i64 && idx < block_end as i64 {
                            let idx = idx as usize;
                            build_candidates_heap_push(
                                &mut old_candidate_priority
                                    [idx * max_candidates..(idx + 1) * max_candidates],
                                &mut old_candidate_indices
                                    [idx * max_candidates..(idx + 1) * max_candidates],
                                d,
                                i as i32,
                            );
                        }
                    }
                }
            }
        }
    }

    // (the reference discards the priority heaps here)
    for i in 0..n_vertices {
        for j in 0..n_neighbors {
            let idx = current_graph.indices[i * n_neighbors + j];

            for k in 0..max_candidates {
                if new_candidate_indices[i * max_candidates + k] == idx {
                    current_graph.flags[i * n_neighbors + j] = 0;
                    break;
                }
            }
        }
    }

    (new_candidate_indices, old_candidate_indices)
}

/// Port of `flagged_heap_push`: like `build_candidates_heap_push` but also
/// maintains a parallel flags array, marking the inserted element with
/// flag 1 ("new").
pub fn flagged_heap_push(
    priorities: &mut [f32],
    indices: &mut [i32],
    flags: &mut [u8],
    p: f32,
    n: i32,
) -> i32 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    // break if we already have this element.
    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    // insert val at position zero
    priorities[0] = p;
    indices[0] = n;

    // descend the heap, swapping values until the max heap criterion is met
    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap;
        if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                i_swap = ic1;
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                i_swap = ic1;
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            i_swap = ic2;
        } else {
            break;
        }

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        flags[i] = flags[i_swap];

        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    flags[i] = 1;

    1
}

/// Port of `apply_graph_update_array`. `update_array` is flat
/// (n_threads x max_updates_per_thread x 3); entry layout `(p, q, d)`
/// stored as f32. Returns the number of successful heap pushes.
pub fn apply_graph_update_array(
    current_graph: &mut NNDHeap,
    update_array: &[f32],
    max_updates_per_thread: usize,
    n_updates_per_thread: &[i32],
    n_threads: usize,
) -> u32 {
    let mut n_changes: u32 = 0;
    let size = current_graph.size;

    let n_vertices = current_graph.n_points;
    let block_size = n_vertices / n_threads + 1;

    for n in 0..n_threads {
        let block_start = n * block_size;
        let block_end = std::cmp::min(block_start + block_size, n_vertices);

        for i in 0..n_threads {
            for j in 0..n_updates_per_thread[i] as usize {
                let base = (i * max_updates_per_thread + j) * 3;
                let p = update_array[base] as i32;

                if p == -1 {
                    break;
                }

                let q = update_array[base + 1] as i32;
                let d = update_array[base + 2];

                if p >= block_start as i32 && p < block_end as i32 {
                    let r = (p as usize) * size..(p as usize + 1) * size;
                    let added = flagged_heap_push(
                        &mut current_graph.distances[r.clone()],
                        &mut current_graph.indices[r.clone()],
                        &mut current_graph.flags[r],
                        d,
                        q,
                    );
                    n_changes += added as u32;
                }

                if q >= block_start as i32 && q < block_end as i32 {
                    let r = (q as usize) * size..(q as usize + 1) * size;
                    let added = flagged_heap_push(
                        &mut current_graph.distances[r.clone()],
                        &mut current_graph.indices[r.clone()],
                        &mut current_graph.flags[r],
                        d,
                        p,
                    );
                    n_changes += added as u32;
                }
            }
        }
    }

    n_changes
}

/// Port of `apply_sorted_graph_updates`: apply pre-sorted graph updates
/// bucketed by target vertex block. `update_array` is flat
/// (n_threads x max_updates x 3); `n_updates_per_block` is flat
/// (n_threads x (n_threads + 1)) with column 0 unused.
pub fn apply_sorted_graph_updates(
    current_graph: &mut NNDHeap,
    update_array: &[f32],
    max_updates: usize,
    n_updates_per_block: &[i32],
    n_threads: usize,
) -> u32 {
    let mut n_changes: u32 = 0;
    let size = current_graph.size;

    let n_vertices = current_graph.n_points;
    let vertex_block_size = n_vertices / n_threads + 1;
    let max_updates_per_thread = max_updates / n_threads;

    for n in 0..n_threads {
        let block_start = n * vertex_block_size;
        let block_end = std::cmp::min(block_start + vertex_block_size, n_vertices);

        // Process all updates in this block's bucket
        // Updates were written by each thread at offset t * max_updates_per_thread
        for t in 0..n_threads {
            let thread_start = t * max_updates_per_thread;
            let thread_count = n_updates_per_block[n * (n_threads + 1) + t + 1];

            for j in 0..thread_count as usize {
                let idx = thread_start + j;
                let base = (n * max_updates + idx) * 3;
                let p = update_array[base] as i32;
                let q = update_array[base + 1] as i32;
                let d = update_array[base + 2];

                // Apply update to p if it's in this block
                if p >= block_start as i32 && p < block_end as i32 {
                    let r = (p as usize) * size..(p as usize + 1) * size;
                    let added = flagged_heap_push(
                        &mut current_graph.distances[r.clone()],
                        &mut current_graph.indices[r.clone()],
                        &mut current_graph.flags[r],
                        d,
                        q,
                    );
                    n_changes += added as u32;
                }

                // Apply update to q if it's in this block
                if q >= block_start as i32 && q < block_end as i32 {
                    let r = (q as usize) * size..(q as usize + 1) * size;
                    let added = flagged_heap_push(
                        &mut current_graph.distances[r.clone()],
                        &mut current_graph.indices[r.clone()],
                        &mut current_graph.flags[r],
                        d,
                        p,
                    );
                    n_changes += added as u32;
                }
            }
        }
    }

    n_changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_heap_initial_values() {
        let heap = make_heap(2, 3);
        assert_eq!(heap.indices, vec![-1; 6]);
        assert!(heap.distances.iter().all(|&d| d == f32::INFINITY));
        assert_eq!(heap.flags, vec![0; 6]);
    }

    #[test]
    fn flagged_heap_push_sets_flag_and_rejects_duplicates() {
        let mut p = vec![f32::INFINITY; 3];
        let mut i = vec![-1i32; 3];
        let mut f = vec![0u8; 3];
        assert_eq!(flagged_heap_push(&mut p, &mut i, &mut f, 0.5, 7), 1);
        // duplicate index rejected
        assert_eq!(flagged_heap_push(&mut p, &mut i, &mut f, 0.4, 7), 0);
        // new index accepted, flag set
        assert_eq!(flagged_heap_push(&mut p, &mut i, &mut f, 0.4, 8), 1);
        assert!(i.contains(&7) && i.contains(&8));
        for (&idx, &flag) in i.iter().zip(f.iter()) {
            if idx >= 0 {
                assert_eq!(flag, 1);
            }
        }
    }

    #[test]
    fn flagged_heap_push_rejects_p_not_below_root() {
        let mut p = vec![1.0f32, 0.5, 0.25];
        let mut i = vec![10, 11, 12];
        let mut f = vec![1u8, 1, 1];
        assert_eq!(flagged_heap_push(&mut p, &mut i, &mut f, 1.0, 99), 0);
        assert_eq!(flagged_heap_push(&mut p, &mut i, &mut f, 2.0, 99), 0);
    }

    #[test]
    fn build_candidates_heap_push_rejects_duplicates() {
        let mut p = vec![f32::INFINITY; 3];
        let mut i = vec![-1i32; 3];
        assert_eq!(build_candidates_heap_push(&mut p, &mut i, 0.9, 4), 1);
        assert_eq!(build_candidates_heap_push(&mut p, &mut i, 0.1, 4), 0);
    }

    #[test]
    fn deheap_sort_sorts_rows_ascending() {
        // note argument order: (indices, distances)
        let mut i = vec![50, 30, 40, 10, 20];
        let mut d = vec![5.0f32, 3.0, 4.0, 1.0, 2.0];
        deheap_sort(&mut i, &mut d, 5);
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(i, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn apply_graph_update_array_stops_at_minus_one_sentinel() {
        let mut heap = make_heap(4, 2);
        // one thread, capacity 3 updates; second entry is a -1 sentinel so
        // the third (valid) entry must NOT be applied.
        let updates = vec![
            0.0, 1.0, 0.5, // (0, 1, 0.5)
            -1.0, -1.0, 0.0, // sentinel
            2.0, 3.0, 0.25, // must be skipped
        ];
        let n_updates = vec![3i32];
        let changes = apply_graph_update_array(&mut heap, &updates, 3, &n_updates, 1);
        assert_eq!(changes, 2); // pushed into rows 0 and 1 only
        assert!(heap.indices[2 * 2..3 * 2].iter().all(|&x| x == -1));
    }

    #[test]
    fn build_candidates_single_vertex_graph() {
        // 1 vertex whose single neighbor slot is empty (-1): no candidates,
        // but the -1 slot matches the -1-filled candidate rows in phase 2,
        // clearing the flag (faithful reference behavior).
        let mut heap = make_heap(1, 1);
        heap.flags[0] = 1;
        let rng: RngState = [0xFFFF; 3];
        let (new_c, old_c) = build_candidates(&mut heap, 2, &rng, 1);
        assert_eq!(new_c, vec![-1, -1]);
        assert_eq!(old_c, vec![-1, -1]);
        assert_eq!(heap.flags[0], 0); // -1 slot matched -1 candidate entry
    }
}
