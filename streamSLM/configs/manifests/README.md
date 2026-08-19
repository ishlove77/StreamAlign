# Training-side manifest convention

`SubwordUnitsDataset` accepts any combination of CSV paths, globs, and
directories. Extraction writes per-shard CSVs under

    <cache_root>/libritts/<split>/manifest_shard{R}_of{W}.csv
    <cache_root>/emilia/400h/manifest_shard{R}_of{W}.csv

so the standard validation/full training input is just a list of those
parent dirs, e.g.

    --manifest <cache>/libritts/train-clean-100 \
               <cache>/libritts/train-clean-360 \
               <cache>/libritts/train-other-500 \
               <cache>/emilia/400h

When `--manifest` points at a directory, all `*.csv` inside are picked up
as shards (sorted). Globs (`<cache>/libritts/train-*`) work too.

No combine step is needed; we deliberately keep one CSV per shard so
re-extraction of a single shard is independent.
