#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ID統合オフライン処理:
- 入力: faces_index.csv（track_id 列）と mapping.csv（from_id,to_id）
- 出力: faces_index_merged.csv（ID統合反映）、mapping_applied.csv（適用ログ）
"""
import csv
import os
import argparse


def load_mapping(path):
    mp = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            try:
                src = int(row[0])
                dst = int(row[1])
            except Exception:
                continue
            mp[src] = dst
    return mp


def canonical(tid, mp):
    seen = set()
    cur = tid
    while cur in mp and cur not in seen:
        seen.add(cur)
        cur = mp[cur]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_csv", required=True, help="faces_index.csv")
    ap.add_argument("--map", dest="map_csv", required=True, help="mapping CSV: from_id,to_id")
    ap.add_argument("--out", dest="output_csv", default=None, help="merged csv 出力先")
    args = ap.parse_args()

    mp = load_mapping(args.map_csv)
    out = args.output_csv or os.path.join(os.path.dirname(args.input_csv), "faces_index_merged.csv")
    applied = os.path.join(os.path.dirname(out), "mapping_applied.csv")

    with open(args.input_csv, newline="", encoding="utf-8") as f_in, \
         open(out, "w", newline="", encoding="utf-8") as f_out, \
         open(applied, "w", newline="", encoding="utf-8") as f_log:
        r = csv.DictReader(f_in)
        fieldnames = r.fieldnames
        if "track_id" not in fieldnames:
            raise RuntimeError("faces_index.csv に track_id 列が必要です")
        w = csv.DictWriter(f_out, fieldnames=fieldnames)
        w.writeheader()
        wl = csv.writer(f_log)
        wl.writerow(["from_id","to_id"])
        for row in r:
            try:
                tid = int(row["track_id"]) if row.get("track_id") not in (None, "") else None
            except Exception:
                tid = None
            if tid is None:
                w.writerow(row)
                continue
            can = canonical(tid, mp)
            if can != tid:
                row["track_id"] = can
                wl.writerow([tid, can])
            w.writerow(row)

    print(f"[SAVE] merged -> {out}")
    print(f"[SAVE] log -> {applied}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse, datetime as dt
from collections import defaultdict

def read_mapping(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row: continue
            if row[0].startswith('#'): continue
            if len(row) < 2: continue
            try:
                old_id = int(row[0]); new_id = int(row[1])
                mapping[old_id] = new_id
            except: pass
    return mapping

def load_entries(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def write_entries(path, rows):
    fields = ["timestamp","camera_id","track_id_local","global_id","age","gender","gender_confidence","entered_side","confidence","source_time","notes"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def aggregate(rows, bucket_minutes):
    agg = defaultdict(lambda: dict(unique=set(), male=0,female=0,unknown_gender=0,
                                   age_teen=0,age_20s=0,age_30s=0,age_40s=0,age_50s_plus=0,age_unknown=0))
    for e in rows:
        t = dt.datetime.fromisoformat(e["timestamp"])
        m = bucket_minutes
        bucket_start = t.replace(minute=(t.minute//m)*m, second=0, microsecond=0)
        key = bucket_start.isoformat()
        d = agg[key]
        try:
            d["unique"].add(int(e["track_id_local"]))
        except:
            d["unique"].add(e["track_id_local"]) 
        g = e.get("gender")
        if g=="male": d["male"]+=1
        elif g=="female": d["female"]+=1
        else: d["unknown_gender"]+=1
        age = e.get("age")
        try:
            age = None if age in ("", None) else int(age)
        except:
            age = None
        if age is None: d["age_unknown"]+=1
        elif age<20: d["age_teen"]+=1
        elif age<30: d["age_20s"]+=1
        elif age<40: d["age_30s"]+=1
        elif age<50: d["age_40s"]+=1
        else: d["age_50s_plus"]+=1
    return agg

def write_agg(path, agg, bucket_minutes):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["bucket_start","bucket_end","unique_visitors","male","female","unknown_gender",
                    "age_teen","age_20s","age_30s","age_40s","age_50s_plus","age_unknown"])
        for k in sorted(agg.keys()):
            start = dt.datetime.fromisoformat(k)
            end = start + dt.timedelta(minutes=bucket_minutes)
            d = agg[k]
            w.writerow([start.isoformat(), end.isoformat(), len(d["unique"]),
                        d["male"], d["female"], d["unknown_gender"],
                        d["age_teen"], d["age_20s"], d["age_30s"], d["age_40s"], d["age_50s_plus"], d["age_unknown"]])

def maybe_merge_faces_index(faces_index, outdir, mapping):
    if not faces_index or not os.path.isfile(faces_index):
        return None
    out_path = os.path.join(outdir, 'faces_index_merged.csv')
    rows = []
    with open(faces_index, 'r', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            tid = r.get('track_id')
            try:
                tid_i = int(tid)
                r['track_id'] = str(mapping.get(tid_i, tid_i))
            except:
                pass
            rows.append(r)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows: w.writerow(r)
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entries', required=True, help='entries_detail.csv')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--mapping', required=True, help='old_id,new_id のCSV')
    ap.add_argument('--faces-index', default=None, help='faces_index.csv（あればtrack_idをマージした索引も出力）')
    ap.add_argument('--bucket-minutes', type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mapping = read_mapping(args.mapping)
    entries = load_entries(args.entries)
    merged = []
    for e in entries:
        ee = dict(e)
        try:
            tid = int(ee.get('track_id_local'))
            ee['track_id_local'] = str(mapping.get(tid, tid))
        except:
            pass
        merged.append(ee)

    detail_out = os.path.join(args.outdir, 'entries_detail_merged.csv')
    write_entries(detail_out, merged)

    agg = aggregate(merged, args.bucket_minutes)
    counts_out = os.path.join(args.outdir, 'counts_15min_merged.csv')
    write_agg(counts_out, agg, args.bucket_minutes)

    faces_out = maybe_merge_faces_index(args.faces_index, args.outdir, mapping)
    print('[SAVE]', detail_out)
    print('[SAVE]', counts_out)
    if faces_out:
        print('[SAVE]', faces_out)

if __name__ == '__main__':
    main()


