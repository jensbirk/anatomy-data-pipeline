import numpy as np, pandas as pd, glob, sys
def arrow_record_sizes(table):
    import pyarrow as pa
    import pyarrow.compute as pc
    sizes = np.zeros(table.num_rows, dtype=np.int64)
    for col in table.columns:
        t = col.type
        if pa.types.is_string(t) or pa.types.is_binary(t):
            sizes += pc.fill_null(pc.binary_length(col), 0).to_numpy() + 4
        elif pa.types.is_large_string(t) or pa.types.is_large_binary(t):
            sizes += pc.fill_null(pc.binary_length(col), 0).to_numpy() + 8
        else:
            sizes += t.bit_width // 8
    return sizes


def c4_arrow_size_counts():
    import pyarrow as pa
    from collections import Counter
    counts = Counter()
    files = sorted(glob.glob("/raid/datasets/c4/converted/arrow/c4-train.*.arrow"))
    for i, f in enumerate(files):
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(f)).read_all()
        values, n = np.unique(arrow_record_sizes(table), return_counts=True)
        counts.update(dict(zip(values.tolist(), n.tolist())))
        if i % 100 == 0:
            print(f"    C4 shard {i + 1}/{len(files)}", flush=True)
    sizes = np.array(sorted(counts), dtype=np.int64)
    return sizes, np.array([counts[v] for v in sizes], dtype=np.int64)



values, counts = c4_arrow_size_counts()
pd.DataFrame({"size_bytes": values, "count": counts}).to_csv("/tmp/c4_arrow_counts.csv.gz", index=False, compression="gzip")
print("records", counts.sum(), "distinct sizes", len(values), "mean", (values*counts).sum()/counts.sum())

