import csv
import numpy as np


FILENAME = 'data-generation.csv'
ORDER=1


# ─── choose highest‐precision available ───────────────────────────────────────
def get_highest_complex_dtype():
    for name in ('complex256','complex192','complex128'):
        try:
            return np.dtype(name)
        except (AttributeError, TypeError):
            continue
    return np.complex128

COMPLEX_DTYPE = get_highest_complex_dtype()

# print with maximum decimal places
np.set_printoptions(precision=17, suppress=False)

# ─── parser ──────────────────────────────────────────────────────────────────
def parse_cgpt_entries(entry_str, size=ORDER*2):
    """
    Parse a GPT_entries string into a (size x size) complex matrix
    using the highest-precision dtype available.
    """
    tokens = entry_str.strip().split()
    vals = []
    
    # case A: alternating real / imag
    if len(tokens) == size*size*2:
        for i in range(0, len(tokens), 2):
            real = float(tokens[i])
            imag_str = tokens[i+1]
            imag = float(imag_str[:-1]) if imag_str.endswith('i') else float(imag_str)
            vals.append(real + 1j*imag)
    
    # case B: each token is "a+bi"
    elif len(tokens) == size*size:
        for tok in tokens:
            tok_py = tok.replace('i','j')
            vals.append(complex(tok_py))
    
    else:
        raise ValueError(f"Unexpected number of tokens ({len(tokens)})")
    
    # build numpy array at highest precision
    arr = np.array(vals, dtype=COMPLEX_DTYPE)
    return arr.reshape((size, size))

def compute_and_print_singulars(objects):
    """
    For each parsed object, compute the singular values of its three GPT matrices,
    print them out in high precision, and return a new list with these values.
    """
    singular_objects = []
    for idx, obj in enumerate(objects):
        shape = obj['shape']
        sv_list = []
        for mat_i, M in enumerate(obj['theoretical_matrices'], start=1):
            # cast to a supported dtype
            M128 = M.astype(np.complex128)
            # compute singular values
            s = np.linalg.svd(M128, compute_uv=False)
            sv_list.append(s)
            # print in the same style as before
            print(f"\n--- Object {idx+1}, Singular values of matrix #{mat_i} for shape '{shape}' ---")
            print(s)
        singular_objects.append({
            'shape': shape,
            'singular_values': sv_list
        })
    print(f"\nComputed singular values for {len(singular_objects)} objects.")
    return singular_objects

def compute_and_print_invariant_ratios(singular_objects):
    """
    From each entry in singular_objects (which must have keys
      'shape' and 'singular_values' where singular_values is [s1, s2, s3]),
    compute for each of the 4 singular values the ratios
      s1/s2, s2/s3, s3/s1,
    yielding 12 numbers per object. Print them and return a list of
    {'shape': shape, 'ratios': [...] }.
    """
    invariant_objects = []
    for idx, obj in enumerate(singular_objects, start=1):
        shape = obj['shape']
        S1, S2, S3 = obj['singular_values']  # each a length-4 array
        ratios = []
        # for each singular‐value index i=0..3 compute the 3 pairwise ratios
        for i in range(len(S1)):
            v1, v2, v3 = S1[i], S2[i], S3[i]
            ratios.extend([v1 / v2, v2 / v3, v3 / v1])
        # print
        print(f"\n--- Object {idx}: shape = '{shape}' — invariant ratios ---")
        for j, r in enumerate(ratios, start=1):
            print(f"  ratio {j:02d}: {r:.6f}")
        invariant_objects.append({'shape': shape, 'ratios': ratios})
    print(f"\nComputed similarity‐invariant ratios for {len(invariant_objects)} objects.")
    return invariant_objects


def compute_and_print_mean_invariants(invariant_objects):
    """
    From each entry in invariant_objects (keys: 'shape' and 'ratios' of length 12),
    group by shape, compute the mean of each of the 12 ratios, print them, and
    return a list of {'shape': shape, 'mean_ratios': [...]}.
    """
    import numpy as np
    from collections import defaultdict

    # 1) Group all ratio‐lists by shape
    grouped = defaultdict(list)
    for obj in invariant_objects:
        grouped[obj['shape']].append(obj['ratios'])

    mean_invariants = []
    # 2) For each shape, compute mean over axis=0
    for shape, ratios_list in grouped.items():
        arr = np.array(ratios_list, dtype=float)   # shape (n_samples, 12)
        means = arr.mean(axis=0)                   # length-12 vector
        # 3) Print
        print(f"\n=== Mean similarity invariants for shape '{shape}' ===")
        for idx, m in enumerate(means, start=1):
            print(f"  mean ratio {idx:02d}: {m:.6f}")
        mean_invariants.append({
            'shape': shape,
            'mean_ratios': means.tolist()
        })

    print(f"\nComputed mean invariants for {len(mean_invariants)} shapes.")
    return mean_invariants


def compute_and_print_distances_to_mean(mean_objs, invariance_objs, target_shape):
    """
    Given:
      mean_objs        - list of dicts {'shape': str, 'mean_ratios': [12 floats]}
      invariance_objs  - list of dicts {'shape': str, 'ratios': [12 floats]}
      target_shape     - the shape name whose mean we compare against
    
    For each entry in invariance_objs, compute the Euclidean distance between its
    'ratios' vector and the mean_ratios of `target_shape`. Print each distance and
    return the list of distances.
    """
    # 1) Look up the mean vector for the target shape
    mean_vec = None
    for m in mean_objs:
        if m['shape'] == target_shape:
            mean_vec = np.array(m['mean_ratios'], dtype=float)
            break
    if mean_vec is None:
        raise ValueError(f"No mean invariants found for shape '{target_shape}'")

    # 2) Compute distances
    distances = []
    print(f"\nDistances of each sample to mean invariants of '{target_shape}':")
    for idx, obj in enumerate(invariance_objs, start=1):
        vec = np.array(obj['ratios'], dtype=float)
        dist = np.linalg.norm(vec - mean_vec)
        distances.append(dist)
        print(f"  Sample {idx:03d} (shape={obj['shape']}):  dist = {dist:.6f}")

    return distances


import csv
from itertools import combinations

def save_mean_invariants_csv(mean_objs, csv_filename):
    """
    Write out the mean invariants per shape to a CSV file.

    Parameters
    ----------
    mean_objs : list of dict
        Each dict must have keys:
          - 'shape': str
          - 'mean_ratios': list of floats (length = n_svs * n_freq_pairs)
    csv_filename : str
        Path to the output CSV.

    The CSV will have columns:
      shape,
      ratio_1_r100-150, ratio_1_r100-200, ratio_1_r150-200,
      ratio_2_r100-150, ratio_2_r100-200, ratio_2_r150-200,
      ...
    """
    # Define the measurement frequencies and all unique pairs
    freqs = [100, 150, 200]
    freq_pairs = list(combinations(freqs, 2))  # [(100,150),(100,200),(150,200)]

    # Determine how many singular‐value ratios per frequency‐pair
    if not mean_objs:
        raise ValueError("mean_objs is empty")
    n_total = len(mean_objs[0]['mean_ratios'])
    n_pairs = len(freq_pairs)
    if n_total % n_pairs != 0:
        raise ValueError("Unexpected number of ratios in mean_objs[0]")
    n_svs = n_total // n_pairs

    # Build header
    header = ['shape']
    for k in range(1, n_svs+1):
        for f1, f2 in freq_pairs:
            header.append(f'ratio_{k}_r{f1}-{f2}')

    # Write CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for obj in mean_objs:
            row = [obj['shape']] + obj['mean_ratios']
            writer.writerow(row)

import numpy as np

def normalize_mean_invariants(mean_objs):
    """
    Z-score normalizes the mean_ratios across shapes.

    Parameters
    ----------
    mean_objs : list of dict
        Each dict must have keys:
          - 'shape': str
          - 'mean_ratios': list or 1D array of floats

    Returns
    -------
    norm_objs : list of dict
        Same as mean_objs but with 'normalized_ratios' instead of 'mean_ratios'.
    """
    if not mean_objs:
        return []

    # Build matrix of shape (n_shapes, n_ratios)
    ratios_matrix = np.stack([obj['mean_ratios'] for obj in mean_objs], axis=0)
    # Compute per-column mean and std
    mu = ratios_matrix.mean(axis=0, keepdims=True)
    sigma = ratios_matrix.std(axis=0, keepdims=True) + 1e-12

    # Z-score normalize
    norm_matrix = (ratios_matrix - mu) / sigma

    # Rebuild list of dicts
    norm_objs = []
    for obj, norm_row in zip(mean_objs, norm_matrix):
        norm_objs.append({
            'shape': obj['shape'],
            'mean_ratios': norm_row.tolist()
        })

    return norm_objs



def main():
    filename =FILENAME
    objects = []

    # read entire CSV
    with open(filename, newline='') as f:
        rows = list(csv.DictReader(f))

    # process in 6-row blocks
    for block_start in range(0, len(rows), 6):
        block = rows[block_start:block_start+6]
        if len(block) < 3:
            break

        theory = block[:3]
        shape = theory[0]['shape']

        mats = []
        for i, row in enumerate(theory, start=1):
            M = parse_cgpt_entries(row['CGPT_entries'], size=2*ORDER)
            mats.append(M)
            #print(f"\n--- Object {len(objects)+1}, Matrix #{i} for shape '{shape}' ---")
            print(M)

        objects.append({
            'shape': shape,
            'theoretical_matrices': mats
        })
        #print(f"Stored object {len(objects)} for shape '{shape}'\n" + "-"*60)

    print(f"\nTotal theoretical objects parsed: {len(objects)}")

    singular_objs = compute_and_print_singulars(objects)
    invariant_objs = compute_and_print_invariant_ratios(singular_objs)
    mean_invariants = compute_and_print_mean_invariants(invariant_objs)
    mean_invariants = normalize_mean_invariants(mean_invariants)
    dists = compute_and_print_distances_to_mean(mean_invariants, invariant_objs, 'circle')
    save_mean_invariants_csv(mean_invariants, 'mean_invariants_by_shape.csv')



if __name__ == '__main__':
    main()
