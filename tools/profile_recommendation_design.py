"""Read-only source profiling for the recommendation design (AGENTS.md, data section).

Writes a separate evidence JSON and never changes source data or experiment models.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 't_ecd_data/dataset/small/retail'
OUTPUT = ROOT / 'results/design_review/source_profile.json'
DAY_US = 86_400_000_000


def main():
    files = sorted((SOURCE / 'events').glob('*.pq'))
    last_day = int(files[-1].stem)
    nulls, actions, contexts, systems = Counter(), Counter(), Counter(), Counter()
    users, items = set(), set()
    rows = mismatch_days = 0
    schemas = Counter()
    item_counts = Counter()
    for index, path in enumerate(files, 1):
        pf = pq.ParquetFile(path)
        schemas[str(pf.schema_arrow)] += 1
        for batch in pf.iter_batches(batch_size=131072):
            rows += batch.num_rows
            for name in batch.schema.names:
                nulls[name] += batch.column(name).null_count
            for name, counts in [('action_type', actions), ('subdomain', contexts), ('os', systems), ('item_id', item_counts)]:
                for v in pc.value_counts(batch.column(name)).to_pylist():
                    counts[v['values']] += v['counts']
            users.update(pc.unique(batch.column('user_id')).to_pylist())
            items.update(pc.unique(batch.column('item_id')).to_pylist())
            timestamps = pc.cast(batch.column('timestamp'), pa.int64())
            mismatch_days += int(np.count_nonzero(timestamps.to_numpy() // DAY_US != int(path.stem)))
        if index % 40 == 0 or index == len(files):
            print(f'Profiled {index}/{len(files)} event files', flush=True)
    catalog_path = SOURCE / 'items.pq'
    catalog = pq.read_table(catalog_path, columns=['item_id', 'brand_id', 'category', 'subcategory', 'price'])
    ids = catalog.column('item_id').to_pylist()
    missing = items - set(ids)
    prices = catalog.column('price').to_numpy()
    subcategory_parents = {}
    for row in catalog.select(['category', 'subcategory']).to_pylist():
        subcategory_parents.setdefault(row['subcategory'], set()).add(row['category'])
    report = {
        'checked_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
        'source': str(SOURCE),
        'snapshot': json.loads((ROOT / 't_ecd_data/retail_snapshot.json').read_text()),
        'event_files': len(files), 'event_rows': rows,
        'source_bytes_including_catalog': sum(p.stat().st_size for p in files) + catalog_path.stat().st_size,
        'event_schemas': dict(schemas), 'event_null_counts': dict(nulls),
        'event_users': len(users), 'event_items': len(items),
        'first_day': int(files[0].stem), 'last_day': last_day,
        'event_day_filename_mismatches': mismatch_days,
        'actions': dict(actions), 'subdomains': dict(contexts), 'operating_systems': dict(systems),
        'catalog_schema': str(pq.read_schema(catalog_path)),
        'catalog_rows': catalog.num_rows,
        'catalog_duplicate_item_ids': len(ids) - len(set(ids)),
        'catalog_null_counts': {n: catalog.column(n).null_count for n in catalog.column_names},
        'catalog_price_negative': int(np.count_nonzero(prices < 0)),
        'catalog_price_nonfinite': int(np.count_nonzero(~np.isfinite(prices))),
        'catalog_price_range': [float(np.nanmin(prices)), float(np.nanmax(prices))],
        'event_items_missing_catalog': len(missing),
        'event_rows_missing_catalog': sum(item_counts[k] for k in missing),
        'subcategories_with_multiple_categories': sum(len(v) > 1 for v in subcategory_parents.values()),
        'duplicate_event_check': 'Not performed: no source event identifier; identical records may be legitimate repeats.',
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('Source profile saved:', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
