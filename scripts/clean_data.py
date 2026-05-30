"""
Data cleaning pipeline for TechBot BD.

Reads:
    data/raw/laptops.csv      (Ryans, scraped)
    data/raw/phones.csv       (Dazzle, scraped)
    data/raw/categories.xlsx  (sub-category classifications + popularity tags)

Writes:
    techbot/data/products.csv     (master CSV used by the runtime)
    data/cleaning_report.md       (audit log for the course report)

Run:
    python scripts/clean_data.py
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.parse
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Master schema (matches techbot/config.py SHOP_TEMPLATE_COLUMNS + extras)
# ---------------------------------------------------------------------------

MASTER_COLUMNS = [
    "name", "category", "sub_category", "brand", "price_bdt",
    "processor", "generation", "ram_gb", "storage",
    "gpu", "gpu_memory", "display_inch", "battery_mah", "camera_mp",
    "url", "shop", "popularity",
]

KNOWN_LAPTOP_BRANDS = {
    "Lenovo", "Asus", "Apple", "Acer", "HP", "MSI", "Dell",
    "Gigabyte", "Microsoft", "Samsung", "LG", "Razer", "Huawei",
}

POPULARITY_TAGS = {
    "🔥 Hot Product": 4,
    "🛍️ Top Selling": 3,
    "😍 Customers Choice": 2,
    "👍 High Demand": 1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_camera_mp(text) -> int:
    """'50MP main + ultrawide' -> 50, 'Dual' -> 0."""
    if not isinstance(text, str):
        return 0
    m = re.search(r"(\d+)\s*MP", text)
    return int(m.group(1)) if m else 0


def search_url(shop: str, name: str, category: str, brand: str) -> str:
    """
    Return a guaranteed-valid catalog URL for the given shop+category.
    Brand-specific URLs vary in slug per shop (Dazzle uses 'iphone' for
    Apple, iQOO casing differs, etc.) and we can't runtime-verify them
    during data prep, so we fall back to the always-valid top-level
    category page. User can filter from there.
    """
    shop_l = (shop or "").lower()
    if shop_l == "ryans":
        return "https://www.ryans.com/category/all-laptop"
    if shop_l == "dazzle":
        return "https://dazzle.com.bd/categories/phones"
    if shop_l == "startech":
        return ("https://www.startech.com.bd/laptop-notebook"
                if category == "laptop"
                else "https://www.startech.com.bd/mobile-phone")
    return ""


def infer_brand_from_name(name: str) -> str:
    """Use first word of product name if it matches a known brand."""
    if not isinstance(name, str) or not name.strip():
        return "Unknown"
    first = name.strip().split()[0]
    return first if first in KNOWN_LAPTOP_BRANDS else "Unknown"


def popularity_from_specs(specs: str) -> int:
    if not isinstance(specs, str):
        return 0
    for tag, score in POPULARITY_TAGS.items():
        if tag in specs:
            return score
    return 0


# ---------------------------------------------------------------------------
# Per-source processors
# ---------------------------------------------------------------------------

def process_phones(path: Path, popularity_lookup: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["category"] = "smartphone"
    df["sub_category"] = "smartphone"  # default; overridden below
    df["camera_mp"] = df["camera_mp"].apply(parse_camera_mp)
    df["url"] = df.apply(
        lambda r: search_url(r["shop"], r["name"], "smartphone", r.get("brand", "")),
        axis=1,
    )

    # Add laptop-only columns as empty placeholders
    df["gpu"] = ""
    df["gpu_memory"] = ""
    df["generation"] = ""

    # Numeric coercion (data is already clean here, but be safe)
    for col in ("ram_gb", "battery_mah", "display_inch", "price_bdt"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["popularity"] = df["name"].map(popularity_lookup).fillna(0).astype(int)
    return df


def process_laptops(path: Path, popularity_lookup: dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["category"] = "laptop"
    df["sub_category"] = "laptop"  # default; overridden below

    # 47 exact duplicates expected from analysis
    before = len(df)
    df = df.drop_duplicates(
        subset=["name", "shop", "price_bdt", "processor", "ram_gb", "storage"]
    ).reset_index(drop=True)
    df.attrs["dropped_dupes"] = before - len(df)

    # Fill 9 missing brands using name first-word
    missing_brand = df["brand"].isna()
    df.attrs["brands_filled"] = int(missing_brand.sum())
    df.loc[missing_brand, "brand"] = df.loc[missing_brand, "name"].apply(infer_brand_from_name)

    # No URL in laptops scrape -> construct shop search URL
    df["url"] = df.apply(
        lambda r: search_url(r["shop"], r["name"], "laptop", r.get("brand", "")),
        axis=1,
    )

    # Add phone-only columns as 0
    df["battery_mah"] = 0
    df["camera_mp"] = 0

    # Generation has 1 NaN -> empty string
    df["generation"] = df["generation"].fillna("")

    df["popularity"] = df["name"].map(popularity_lookup).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# XLSX-driven enrichment
# ---------------------------------------------------------------------------

def load_subcategory_lookups(xlsx_path: Path) -> dict[str, set[str]]:
    """Return name-sets for each sub-category sheet."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    lookups = {}
    for sheet_name in ("Gaming Phone", "Flagship Phone", "Gaming Laptop", "Lightweight Laptop"):
        if sheet_name in sheets:
            names = sheets[sheet_name]["Product Name"].astype(str).str.strip()
            lookups[sheet_name] = set(names)
        else:
            lookups[sheet_name] = set()
    return lookups


def load_popularity_lookup(xlsx_path: Path) -> dict[str, int]:
    """Pull review-tag popularity from Phone sheet specs column."""
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    lookup = {}
    for sheet_name in ("Phone", "Laptop"):
        if sheet_name not in sheets:
            continue
        sheet = sheets[sheet_name]
        if "Specs" not in sheet.columns:
            continue
        for _, row in sheet.iterrows():
            name = str(row["Product Name"]).strip()
            score = popularity_from_specs(row["Specs"])
            if score > lookup.get(name, 0):
                lookup[name] = score
    return lookup


def assign_subcategory(df: pd.DataFrame, lookups: dict[str, set[str]]) -> tuple[pd.DataFrame, dict]:
    """Apply most-specific-wins sub-category logic. Returns (df, stats)."""
    stats = {"gaming_phone": 0, "flagship_phone": 0, "gaming_laptop": 0,
             "lightweight_laptop": 0, "default_phone": 0, "default_laptop": 0}

    def assign(row):
        name = row["name"]
        if row["category"] == "smartphone":
            if name in lookups["Gaming Phone"]:
                stats["gaming_phone"] += 1
                return "gaming_phone"
            if name in lookups["Flagship Phone"]:
                stats["flagship_phone"] += 1
                return "flagship_phone"
            stats["default_phone"] += 1
            return "smartphone"
        # laptop
        if name in lookups["Gaming Laptop"]:
            stats["gaming_laptop"] += 1
            return "gaming_laptop"
        if name in lookups["Lightweight Laptop"]:
            stats["lightweight_laptop"] += 1
            return "lightweight_laptop"
        stats["default_laptop"] += 1
        return "laptop"

    df["sub_category"] = df.apply(assign, axis=1)
    return df, stats


# ---------------------------------------------------------------------------
# Validation + write
# ---------------------------------------------------------------------------

ALLOWED_SUBCATS = {"laptop", "smartphone", "gaming_laptop", "lightweight_laptop",
                   "gaming_phone", "flagship_phone", "budget_phone"}

def validate(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in ("name", "category", "price_bdt", "shop"):
        if df[col].isna().any():
            errors.append(f"Nulls present in required column '{col}'")
    bad_cat = set(df["category"].unique()) - {"laptop", "smartphone"}
    if bad_cat:
        errors.append(f"Invalid category values: {bad_cat}")
    bad_sub = set(df["sub_category"].unique()) - ALLOWED_SUBCATS
    if bad_sub:
        errors.append(f"Invalid sub_category values: {bad_sub}")
    if (df["price_bdt"] <= 0).any():
        errors.append(f"{(df['price_bdt'] <= 0).sum()} rows with non-positive price")
    if (df["price_bdt"] > 1_000_000).any():
        errors.append(f"{(df['price_bdt'] > 1_000_000).sum()} rows with implausibly high price")
    return errors


def write_report(out: Path, stats: dict) -> None:
    lines = ["# TechBot BD — Data Cleaning Report\n"]
    for k, v in stats.items():
        lines.append(f"- **{k}**: {v}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptops", default="data/raw/laptops.csv")
    ap.add_argument("--phones", default="data/raw/phones.csv")
    ap.add_argument("--xlsx", default="data/raw/categories.xlsx")
    ap.add_argument("--out", default="techbot/data/products.csv")
    ap.add_argument("--report", default="data/cleaning_report.md")
    args = ap.parse_args()

    laptops_path = Path(args.laptops)
    phones_path = Path(args.phones)
    xlsx_path = Path(args.xlsx)

    print("[1/6] Loading XLSX lookups...")
    subcat_lookups = load_subcategory_lookups(xlsx_path)
    popularity_lookup = load_popularity_lookup(xlsx_path)
    print(f"      sub-cats: {[(k, len(v)) for k, v in subcat_lookups.items()]}")
    print(f"      popularity entries: {len(popularity_lookup)}")

    print("[2/6] Processing phones...")
    phones = process_phones(phones_path, popularity_lookup)
    print(f"      {len(phones)} rows")

    print("[3/6] Processing laptops...")
    laptops = process_laptops(laptops_path, popularity_lookup)
    print(f"      {len(laptops)} rows (dropped {laptops.attrs.get('dropped_dupes', 0)} dupes, "
          f"filled {laptops.attrs.get('brands_filled', 0)} brands)")

    print("[4/6] Concatenating + assigning sub_category...")
    merged = pd.concat([phones, laptops], ignore_index=True)
    # Keep only master columns
    for col in MASTER_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[MASTER_COLUMNS]
    merged, subcat_stats = assign_subcategory(merged, subcat_lookups)

    print("[5/6] Validating...")
    errors = validate(merged)
    if errors:
        print("      VALIDATION FAILURES:")
        for e in errors:
            print(f"        - {e}")
        # Don't crash; write anyway and let the user see
    else:
        print("      OK")

    print(f"[6/6] Writing {args.out} ({len(merged)} rows)...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    stats = {
        "total_rows": len(merged),
        "phones": int((merged["category"] == "smartphone").sum()),
        "laptops": int((merged["category"] == "laptop").sum()),
        "shops": ", ".join(sorted(merged["shop"].unique())),
        "laptop_dupes_dropped": laptops.attrs.get("dropped_dupes", 0),
        "laptop_brands_inferred": laptops.attrs.get("brands_filled", 0),
        **subcat_stats,
        "validation_errors": errors or "none",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_report(Path(args.report), stats)

    print("\nDone.")
    print(f"  Output:  {args.out}")
    print(f"  Report:  {args.report}")
    print(f"  Subcat distribution: {subcat_stats}")


if __name__ == "__main__":
    main()
