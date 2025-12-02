import pandas as pd
import numpy as np
import pickle
import random
import zlib
import itertools
from mpi4py import MPI
from collections import defaultdict
import sys
import os


# HPC CONFIGURATION
INPUT_FILE = "GlobalLandTemperaturesByCity.csv"
OUTPUT_FILE = "final_verified_matches.pkl"

# Algorithm Parameters
NUM_HASHES = 200
BANDS = 20
ROWS = 10 
LARGE_PRIME = (2**61) - 1
JACCARD_THRESHOLD = 0.80

# KRAKEN Mode settings
MAX_BUCKET_SIZE = 50             # this helps in preventing Tropical infinite loops
MAX_RESULTS_PER_WORKER = 100000


def clean_coordinate(coord_str):
    if pd.isna(coord_str): return np.nan
    coord_str = str(coord_str).strip()
    if coord_str[-1] in ['N', 'S', 'E', 'W']:
        num = float(coord_str[:-1])
        if coord_str[-1] in ['S', 'W']:
            num = -num
        return num
    return float(coord_str)

def discretize_temperature(temp):
    """
    SCIENTIFIC FEATURE 1: High Resolution
    Mapping float temp to 2-degree buckets.
    """
    if pd.isna(temp): return "Unknown"
    lower_bound = int(temp // 2) * 2
    return f"{lower_bound}"

def step_1_data_cleaning(file_path):
    print(f"[Master] Loading and cleaning {file_path}...", flush=True)
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['AverageTemperature'])
    df['dt'] = pd.to_datetime(df['dt'])
    df['Year'] = df['dt'].dt.year
    df['Month'] = df['dt'].dt.month
    df['Latitude_Float'] = df['Latitude'].apply(clean_coordinate)
    df['Longitude_Float'] = df['Longitude'].apply(clean_coordinate)

    pivot_df = df.pivot_table(
        index=['City', 'Country', 'Year', 'Latitude_Float', 'Longitude_Float'],
        columns='Month',
        values='AverageTemperature'
    ).reset_index()

    month_map = {i: f"{m}_Temp" for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}
    pivot_df = pivot_df.rename(columns=month_map)
    pivot_df = pivot_df.dropna(subset=month_map.values())
    pivot_df['City_Year_ID'] = pivot_df['City'] + '-' + pivot_df['Year'].astype(str)
    
    print(f"[Master] Data Cleaned. Total City-Years: {len(pivot_df)}", flush=True)
    return pivot_df

def step_2_shingling(df):
    """
    [SCIENTIFIC FEATURE 2: Trend Detection]
    Captures both absolute temperature AND the shape of the curve (Rise/Fall).
    """
    print("[Master] Generating Shingles (Temp + Trend)...", flush=True)
    output_list = []
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_cols = [f"{m}_Temp" for m in months]

    for row in df.itertuples(index=False):
        shingle_set = set()
        monthly_values = []
        
        for m_name, col in zip(months, month_cols):
            val = getattr(row, col)
            monthly_values.append(val)
            # 1. Absolute Temp Shingle
            label = discretize_temperature(val)
            shingle_set.add(f"{m_name}-{label}")
        
        # 2. Trend Shingles
        for i in range(1, 12): 
            prev_temp = monthly_values[i-1]
            curr_temp = monthly_values[i]
            current_month_name = months[i]
            
            if pd.isna(prev_temp) or pd.isna(curr_temp):
                trend = "Unknown"
            else:
                diff = curr_temp - prev_temp
                if diff > 1.0: trend = "Rise"
                elif diff < -1.0: trend = "Fall"
                else: trend = "Stable"
            
            shingle_set.add(f"{current_month_name}-Trend-{trend}")

        output_list.append((row.City_Year_ID, shingle_set, row.Latitude_Float, row.Longitude_Float))
    return output_list

def get_hash_functions(n=200):
    coeffs = []
    for _ in range(n):
        a = random.randint(1, LARGE_PRIME - 1)
        if a % 2 == 0: a += 1
        b = random.randint(0, LARGE_PRIME - 1)
        coeffs.append((a, b))
    return coeffs

def string_to_int(s):
    return zlib.crc32(s.encode('utf-8')) & 0xffffffff

def compute_minhash_signature(shingle_set, hash_coeffs):
    signature = []
    integer_shingles = [string_to_int(s) for s in shingle_set]
    for a, b in hash_coeffs:
        min_h = float('inf')
        for x in integer_shingles:
            phash = (a * x + b) % LARGE_PRIME
            if phash < min_h: min_h = phash
        signature.append(min_h)
    return signature

def step_6_banding_local(signature_list, bands=20, rows=10):
    bucket_list = []
    for city_id, signature in signature_list:
        for band_idx in range(bands):
            start = band_idx * rows
            end = start + rows
            band_vec = tuple(signature[start:end])
            deterministic_str = f"{band_idx}-{str(band_vec)}"
            b_id = zlib.crc32(deterministic_str.encode('utf-8')) & 0xffffffff
            bucket_list.append((b_id, city_id))
    return bucket_list

def step_8_verification_parallel(local_buckets, original_data_dict):
    verified_matches = []
    checked_pairs = set()

    for bucket_id, city_list in local_buckets:
        if len(city_list) > MAX_BUCKET_SIZE:
            continue
            
        n = len(city_list)
        for i in range(n):
            for j in range(i + 1, n):
                city_a = city_list[i]
                city_b = city_list[j]
                
                pair_key = tuple(sorted((city_a, city_b)))
                if pair_key in checked_pairs: continue
                checked_pairs.add(pair_key)
                
                if city_a not in original_data_dict or city_b not in original_data_dict:
                    continue

                # FEATURE 3: Teleconnection Filter
                # Ignoring neighbors within 10 degrees
                lat_a = original_data_dict[city_a]['lat']
                long_a = original_data_dict[city_a]['long']
                lat_b = original_data_dict[city_b]['lat']
                long_b = original_data_dict[city_b]['long']

                if abs(lat_a - lat_b) < 10 and abs(long_a - long_b) < 10:
                    continue

                set_a = original_data_dict[city_a]['shingles']
                set_b = original_data_dict[city_b]['shingles']
                
                inter = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                sim = inter / union if union > 0 else 0
                
                if sim >= JACCARD_THRESHOLD:
                    verified_matches.append({
                        "source": city_a,
                        "target": city_b,
                        "sim": round(sim, 4),
                        "src_coord": (lat_a, long_a),
                        "tgt_coord": (lat_b, long_b)
                    })
                    
                if len(verified_matches) >= MAX_RESULTS_PER_WORKER:
                    return verified_matches
                    
    return verified_matches


# Main Execution Function
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Master preparation
    if rank == 0:
        print(f"--- HPC MPI Engine Started with {size} Cores (KRAKEN MODE) ---", flush=True)
        if not os.path.exists(INPUT_FILE):
            print(f"CRITICAL ERROR: {INPUT_FILE} not found.", flush=True)
            sys.exit(1)
            
        df_clean = step_1_data_cleaning(INPUT_FILE)
        full_data_list = step_2_shingling(df_clean)
        
        original_data_dict = {
            item[0]: {"shingles": item[1], "lat": item[2], "long": item[3]}
            for item in full_data_list
        }
        
        print("[Master] Generating Hash Functions...", flush=True)
        hash_coeffs = get_hash_functions(NUM_HASHES)
        
        print(f"[Master] Scattering {len(full_data_list)} cities to {size} workers...", flush=True)
        chunks = np.array_split(full_data_list, size)
    else:
        full_data_list = None
        hash_coeffs = None
        chunks = None
        original_data_dict = None

    # PHASE 1: PARALLEL HASHING
    hash_coeffs = comm.bcast(hash_coeffs, root=0)
    local_data = comm.scatter(chunks, root=0)

    print(f"[Worker {rank}] Processing {len(local_data)} cities...", flush=True)
    local_signatures = []
    for city_id, shingle_set, lat, long in local_data:
        sig = compute_minhash_signature(shingle_set, hash_coeffs)
        local_signatures.append((city_id, sig))
        
    local_bucket_pairs = step_6_banding_local(local_signatures, bands=BANDS, rows=ROWS)
    
    del local_signatures
    del local_data

    # PHASE 2: GATHERING BUCKETS
    all_buckets_nested = comm.gather(local_bucket_pairs, root=0)

    #PHASE 3: REDISTRIBUTION
    bucket_chunks = None
    if rank == 0:
        print("[Master] Aggregating buckets in RAM...", flush=True)
        flat_bucket_list = [item for sublist in all_buckets_nested for item in sublist]
        del all_buckets_nested 

        buckets = defaultdict(list)
        for b_id, city_id in flat_bucket_list:
            buckets[b_id].append(city_id)
        
        candidate_buckets_list = [(k, v) for k, v in buckets.items() if len(v) > 1]
        print(f"[Master] Found {len(candidate_buckets_list)} buckets to verify.", flush=True)
        
        print("[Master] Scattering buckets for Verification...", flush=True)
        chunk_size = len(candidate_buckets_list) // size
        bucket_chunks = []
        for i in range(size):
            start = i * chunk_size
            if i == size - 1:
                bucket_chunks.append(candidate_buckets_list[start:])
            else:
                bucket_chunks.append(candidate_buckets_list[start:start + chunk_size])

    # PHASE 4: PARALLEL VERIFICATION
    original_data_dict = comm.bcast(original_data_dict, root=0)
    local_buckets = comm.scatter(bucket_chunks, root=0)
    
    print(f"[Worker {rank}] Verifying {len(local_buckets)} buckets...", flush=True)
    local_matches = step_8_verification_parallel(local_buckets, original_data_dict)
    print(f"[Worker {rank}] Found {len(local_matches)} matches.", flush=True)
    
    # FEATURE 4: THE UNLEASHING
    # Here removed the [:5000] slice
    local_matches.sort(key=lambda x: x['sim'], reverse=True)
    

    temp_file = f"temp_matches_rank_{rank}.pkl"
    with open(temp_file, 'wb') as f:
        pickle.dump(local_matches, f)
    
    comm.barrier()

    # Saving Files
    if rank == 0:
        print("[Master] Collecting results from all workers...", flush=True)
        final_results = []
        for i in range(size):
            temp_file = f"temp_matches_rank_{i}.pkl"
            with open(temp_file, 'rb') as f:
                worker_matches = pickle.load(f)
                final_results.extend(worker_matches)
            os.remove(temp_file)
        
        print(f"\nSUCCESS: Found {len(final_results)} verified matches.", flush=True)
        
        final_results.sort(key=lambda x: x['sim'], reverse=True)
        
        if len(final_results) > 0:
            print("Top 3 Matches:", flush=True)
            for m in final_results[:3]:
                print(f"  {m['source']} <-> {m['target']} (Sim: {m['sim']})", flush=True)
        
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"[Master] Results saved to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()