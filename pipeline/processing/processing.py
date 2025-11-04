import pandas as pd
import numpy as np
import pprint
from pathlib import Path
from rapidfuzz import process
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent
STATE_ROOT = BASE_DIR / "States"
COUNTY_LIST_ROOT = BASE_DIR / "CountyList"
PROCESSED_ROOT = BASE_DIR / "Processed_Data"
STATE = "AL"
TARGET_DATE_STR = "2024-05-23"
TARGET_DATE = pd.to_datetime(TARGET_DATE_STR).date()

### PREPROCESSING WORK

# Get all column names from every provider CSV
files = list(STATE_ROOT.rglob("*.csv"))
cols = set()   

for file in files:
    df = pd.read_csv(file, nrows=0)                            
    df.columns = [col.strip().lower() for col in df.columns]    
    cols.update(df.columns)

county = ['countynam', 'countyname', 'county', 'counties', 'area', 'name', 'area_name']
c_affected = ['out', 'customersaffected', '# out', 'n_out', 'aff', 'cust_a', 'customersoutnow', 'numoutages']
c_served = ['served', 'customersserved', '# served', 'premisecount', 'cust_s', 'total', 'customercount', 'count']
timestamp = ['timestamp']
col_map = {}

for name in county:
    col_map[name] = "county"
for caff in c_affected:
    col_map[caff] = "customers_affected"
for cser in c_served:
    col_map[cser] = "customers_served"
for time in timestamp:
    col_map[time] = "timestamp"

# Get the raw county set for each state
raw_county_dict = {}

for state_path in STATE_ROOT.glob("*"):
    if not state_path.is_dir():
        continue
    state_code = state_path.name.upper()
    county_set = set()

    for file in state_path.glob("*.csv"):
        df = pd.read_csv(file)
        df.columns = [col.strip().lower() for col in df.columns]
        col_name = [col for col in df.columns if col in county]
        key_col = [col for col in df.columns if col == 'key']

        # Some providers also include municipalities, this filters those out
        if 'key' in df.columns:
            mask = df['key'].str.lower() != 'muni'
            county_set.update(c.strip().lower() for c in df.loc[mask, col_name[0]].dropna())
        else:
            county_set.update(c.strip().lower() for c in df[col_name[0]].dropna())
        
    county_set = sorted(county_set)
    raw_county_dict[state_code] = county_set

# Compile the "master" county list we will use
master_county_dict = {}

for state_file in COUNTY_LIST_ROOT.glob("*.txt"):
    with state_file.open('r', encoding="utf-8") as file:
        state_code = state_file.stem.upper()
        lines = file.readlines()
        county_list = []
        for line in lines:
            county_list.append(line.lower().replace("county", "").strip())
        master_county_dict[state_code] = county_list

# Check for different names being used for the same county
dupe_dict = {}  # Stores {State : {original name : final name}}
dupe_list = {}  # Stores {State : [original names]} 

for code, raw_names in raw_county_dict.items():
    master_names = master_county_dict.get(code)
    if not master_names:
        dupe_dict[code] = {}
        dupe_list[code] = []
        continue
    dupe_entry = {}
    dupe_names = []

    for raw in raw_names:
        match, score, _ = process.extractOne(raw, master_names)
        if score >= 85: 
            dupe_entry[raw] = match
	    # dupe_entry.append({raw: {match: score}})
            dupe_names.append(raw)
    dupe_dict[code] = dupe_entry
    dupe_list[code] = dupe_names

# Standardizes the df columns for processing purposes
def standardize_cols(df, state):
    # Remove duplicate customers affected columns, keeping the one with the highest sum
    dupes = [c for c in df.columns if c.strip().lower() in c_affected]

    if len(dupes) > 1:
        keep = df[dupes].sum(skipna=True).idxmax()
        df.drop(columns=[c for c in df.columns if c != keep and c in dupes], inplace=True)

    # Standardize column names and remove unnecessary columns
    for col in df.columns:
        lower = col.strip().lower()

        if lower in col_map:
            df.rename(columns={col: col_map[lower]}, inplace=True)
        elif lower != "% out":
            df.drop(columns=[col], inplace=True)
    
    # Remove invalid rows in county col
    df = df.dropna(subset=['county'])
    df = df[~df['county'].isin(['unknown', 'Unknown', 'UNKNOWN', ''])].copy()

    # Calculate estimated customers served using affected / % out data
    if "% out" in df.columns:
        if "customers_served" in df.columns:
            df.drop(columns="% out", inplace=True)
        else:
            df['% out'] = df['% out'].str.replace("<", "", regex=False).str.rstrip('%').replace("", np.nan).astype(float)
            df['% out'] = pd.to_numeric(df['% out'], errors="coerce") / 100
            df['customers_served'] = np.where(
                (df['% out'].notna()) & (df['% out'] != 0), 
                ((df['customers_affected'].astype(float) / df['% out']).round().astype("Int64")), 
                np.nan)
            df.drop(columns=['% out'], inplace=True)

    # Check for if any columns are still missing. If so, skip this provider
    std_names = ['county', 'customers_affected', 'customers_served', 'timestamp']
    skip = False
    missing = []

    for name in std_names:
        if name not in df.columns:
            missing.append(name)
            skip = True
        
    if skip:
        return [False, df]  
    else:
        # Standardize customers affected and served columns (e.g. "123,456" --> 123456)
        df['customers_affected'] = (
            pd.to_numeric(
                df['customers_affected']
                    .astype(str)
                    .str.replace('"', '', regex=False)
                    .str.replace(',', '', regex=False),
                    errors='coerce'
            ).fillna(0)
        )
        
        df['customers_served'] = (
            pd.to_numeric(
                df['customers_served']
                    .astype(str)
                    .str.replace('"', '', regex=False)
                    .str.replace(',', '', regex=False),
                    errors='coerce'
            ).fillna(0)
        )
        
        # Standardize county names
        county_dict = dupe_dict.get(state, {})
        df = df.copy()
        standardized_county = df['county'].astype(str).str.strip().str.lower()
        if county_dict:
            df['county'] = standardized_county.map(county_dict)
            df = df[df['county'].map(type) == str]
        else:
            df['county'] = standardized_county
        return [True, df]

# Pull most recent customers served data as a baseline

# ---------------- NEW SECTION ----------------
# Build county → [EMC, customers_served, last_updated] mapping and historical served CSV

STATE_DIR = STATE_ROOT / STATE
SUMMARY_PATH = PROCESSED_ROOT / f"{STATE}_customers_served_summary.csv"

# Step 1: Gather all per-county-per-provider data
records = []

for file in STATE_DIR.glob("*.csv"):
    provider = file.name.replace("per_county_", "").replace(".csv", "")
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    res = standardize_cols(df, STATE)
    if not res[0]:
        continue
    df = res[1]

    # Clean timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].dt.date == TARGET_DATE]
    if df.empty:
        continue

    df["customers_served"] = pd.to_numeric(df["customers_served"], errors="coerce")
    df = df.dropna(subset=["customers_served"])
    if df.empty:
        continue

    # Group by county and keep the largest served count with the most recent timestamp
    provider_summary = (
        df.sort_values(by=["county", "customers_served", "timestamp"], ascending=[True, False, False])
          .groupby("county", as_index=False)
          .first()
    )
    provider_summary["emc"] = provider
    provider_summary = provider_summary[["county", "emc", "customers_served", "timestamp"]]

    records.append(provider_summary)

emc_max = {}
served_dict = {}
county_totals = {}
if not records:
    print(f"No valid files found for {STATE}")
else:
    all_data = pd.concat(records, ignore_index=True)
    all_data = all_data.dropna(subset=["county", "customers_served"])
    all_data["customers_served"] = pd.to_numeric(all_data["customers_served"], errors="coerce").fillna(0)
    all_data = (
        all_data.sort_values(["county", "emc", "customers_served", "timestamp"], ascending=[True, True, False, False])
                .groupby(["county", "emc"], as_index=False)
                .first()
    )

    # Step 2: Capture per-EMC maxima and county totals
    for _, row in all_data.iterrows():
        county = row["county"]
        emc = row["emc"]
        served = int(round(float(row["customers_served"])))
        ts = row["timestamp"]
        emc_max[(county, emc)] = {
            "customers_served": served,
            "last_updated": ts,
        }
        served_dict.setdefault(county, []).append([emc, served, ts])

    for county in served_dict:
        served_dict[county].sort(key=lambda rec: rec[0])

    county_totals = {
        county: sum(rec[1] for rec in recs)
        for county, recs in served_dict.items()
    }
    served_dict = dict(sorted(served_dict.items()))
    total_customers_served = sum(county_totals.values())

    # Step 3: Mapping dictionary {county: [emc, customers_served, last_updated]}
    # (served_dict already populated in desired format)

    # Step 4: Load previous summary (if exists) and compare changes
    update_needed = True
    if SUMMARY_PATH.exists():
        prev_df = pd.read_csv(SUMMARY_PATH)
        prev_df.columns = (
            prev_df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        if {"county", "emc"}.issubset(prev_df.columns):
            merged = pd.merge(
                all_data,
                prev_df,
                on=["county", "emc"],
                suffixes=("", "_old"),
                how="outer"
            )
            if "customers_served_old" in merged.columns:
                merged["customers_served"] = pd.to_numeric(
                    merged["customers_served"], errors="coerce"
                )
                merged["customers_served_old"] = pd.to_numeric(
                    merged["customers_served_old"], errors="coerce"
                )
                merged["change_pct"] = abs(
                    merged["customers_served"] - merged["customers_served_old"]
                ) / merged["customers_served_old"].replace(0, np.nan)

                if merged["change_pct"].max(skipna=True) < 0.05:
                    update_needed = False
                    print("No significant changes (>5%) detected; CSV not updated.")
                else:
                    print("Significant change detected; updating CSV...")
            else:
                print("Previous summary missing customers served values; regenerating CSV.")
        else:
            print("Previous summary missing merge keys; regenerating CSV.")

    # Step 5: Save to CSV
    if update_needed:
        PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
        all_data.rename(
            columns={
                "county": "County",
                "emc": "EMC",
                "customers_served": "Customers Served",
                "timestamp": "Last Updated"
            },
            inplace=True,
        )
        all_data.to_csv(SUMMARY_PATH, index=False)
        print(f"✅ Saved summary CSV: {SUMMARY_PATH}")
    else:
        PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    totals_path = PROCESSED_ROOT / f"{STATE}_county_totals_{TARGET_DATE_STR}.csv"
    totals_records = []
    for county in sorted(master_county_dict.get(STATE, [])):
        total = int(round(county_totals.get(county, 0)))
        totals_records.append({"County": county, "Total Customers Served": total})
    totals_df = pd.DataFrame(totals_records)
    totals_df.to_csv(totals_path, index=False)
    print(f"✅ Saved county totals CSV: {totals_path}")

    # Step 6: Print summary
    print(f"Total customers served (sum of EMC maxima per county): {total_customers_served:,.0f}")
    print("County → [EMC, customers_served_max_for_that_EMC, last_updated]:")
    pprint.pprint(served_dict)


# for state in glob.glob(os.path.join(base, '*')):
#     state_code = state[-2:]
#     served_dict = {}  # {county : [customers served, last updated date]}

#     for file in glob.glob(os.path.join(state, '*.csv')):
#         # Standardize cols 
#         df = pd.read_csv(file)
#         df.columns = df.columns.str.strip().str.lower()
#         res = standardize_cols(df, state_code)
#         if res[0]:
#             df = res[1]
#         else:
#             continue

#         # Group by county name and sort by timestamp
#         df = df.sort_values(by=['county', 'timestamp'], ascending=[True, False])
#         latest = df.groupby('county', as_index=False).last()
#         latest['customers_served'] = pd.to_numeric(latest['customers_served'], errors='coerce')
#         latest = latest[latest['customers_served'].notna() & (latest['customers_served'] >= 0)] 

#         # Get the most recent customers served data for each provider
#         for _, row in latest.iterrows():
#             county = row['county']
#             served = row['customers_served']
#             timestamp = row['timestamp']
            
#             if pd.notna(served):
#                 if county in served_dict:
#                     prev_served, prev_ts = served_dict[county]
#                     new_served = served + prev_served
#                     new_ts = max(prev_ts, timestamp)
#                     served_dict[county] = [new_served, new_ts]
#                 else:
#                     served_dict[county] = [served, timestamp]
    
#     df = pd.DataFrame.from_dict(
#         served_dict,
#         orient="index",
#         columns=['customers_served', 'timestamp']
#     ).reset_index()
#     df = df.rename(columns={'index': 'county'})
#     df = df.sort_values('county')
#     # Uncomment below to write data into state csv
#     # df.to_csv(os.path.join("CustomersServed", f"{state_code}_customers_served.csv"))

### PROCESSING 
# Initialize a list of dictionaries, each entry representing a county and its data 
schema = ["ID", "county", "customers_affected", "customers_served", "lower_bound_customers_affected", "start_time", "end_time", "duration"]
state_counties = master_county_dict.get(STATE)
if not state_counties:
    raise KeyError(f"State {STATE} not found in county list directory {COUNTY_LIST_ROOT}")

county_dfs = {c: pd.DataFrame(columns=schema) for c in state_counties}
STATE_DIR = STATE_ROOT / STATE
files = list(STATE_DIR.glob("*.csv"))
date = TARGET_DATE_STR
county_list = dupe_list.get(STATE)
if county_list is None:
    raise KeyError(f"No county mapping found for state {STATE} in dupe list.")

for file in files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()
    
    # Use standardize_cols function
    res = standardize_cols(df, STATE)
    if res[0]:
        df = res[1]
    else:
        continue
    
    # Standardize timestamp data type
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df[df['timestamp'].dt.date == TARGET_DATE]
    # Sort by county name then by date and reorder columns 

    # Sort by county name then by date
    for county in county_list:
        county_df = df[df['county'] == county]
        
        if county_df.size != 0:
            county_df = county_df[['county', 'customers_affected', 'customers_served', 'timestamp']]
            county_df = county_df.sort_values('timestamp')

            # Group by timestamp and give IDs to each new outage
            last_id = county_dfs[county]['ID'].max() if not county_dfs[county].empty else 0
            threshold = timedelta(hours=1, minutes=14)

            county_df['diff'] = county_df['timestamp'].diff()
            mask = county_df['diff'] > threshold
            county_df['new_outage'] = (county_df['diff'].isna() | mask)
            county_df['ID'] = county_df['new_outage'].cumsum() + last_id

            result = (
                county_df.groupby('ID')
                        .agg(
                            county=('county', 'first'),
                            customers_affected=('customers_affected', 'max'),
                            customers_served=('customers_served', 'max'),
                            start_time=('timestamp', 'min'),
                            end_time=('timestamp', 'max')
                        )
                        .reset_index()
            )

            result['lower_bound_customers_affected'] = 0
            result['duration'] = result['end_time'] - result['start_time']

            if county_dfs[county].empty:
                county_dfs[county] = result
            else:
                county_dfs[county] = pd.concat([county_dfs[county], result], ignore_index=True)

# Within each county, sort by chronological starting time
county_customers_served = {}

for county, df_county in list(county_dfs.items()):
    if not df_county.empty:
        df_county = df_county.copy()
        # Convert customers_served to numeric
        df_county['customers_served'] = pd.to_numeric(
            df_county['customers_served'], 
            errors='coerce'
        ).fillna(0)
        
        total_served = county_totals.get(county)
        if total_served is None:
            total_served = int(round(df_county['customers_served'].sum()))
        county_customers_served[county] = int(total_served)
        df_county['customers_served'] = int(total_served)
        county_dfs[county] = df_county
    else:
        total_served = county_totals.get(county, 0)
        county_customers_served[county] = int(total_served)

for county in county_dfs:
    if not county_dfs[county].empty:
        # Convert customers_affected to numeric
        county_dfs[county]['customers_affected'] = pd.to_numeric(
            county_dfs[county]['customers_affected'], 
            errors='coerce'
        ).fillna(0)
        
        if county_dfs[county]['customers_affected'].sum() > 0:
            # Get the maximum customers_affected across ALL outages in this county for the day
            daily_max = county_dfs[county]['customers_affected'].max()
            # Apply this same value to all rows
            county_dfs[county]['lower_bound_customers_affected'] = daily_max

# Create filler dataframes for counties with no reported outages
for county in master_county_dict[STATE]:
    if county_dfs[county].empty:
        result = pd.DataFrame([{
            "ID": 1,
            "county": county,
            "customers_affected": 0,
            "customers_served": county_customers_served.get(county, 0),
            "lower_bound_customers_affected": 0,
            "start_time": pd.NaT,
            "end_time": pd.NaT,
            "duration": pd.Timedelta(0)
        }], columns=schema) 
        county_dfs[county] = result
        
# Uncomment below to print county level data 
pprint.pprint(county_dfs)

# Write data to CSV
combined = pd.concat(county_dfs.values(), ignore_index=True)
combined['customers_served'] = pd.to_numeric(combined['customers_served'], errors='coerce').fillna(0).astype(int)
# Uncomment below to write data to corresponding csv file
combined.to_csv(PROCESSED_ROOT / f"{STATE}_all_counties_{date}.csv", index=False)

num_counties = combined['county'].nunique()
print(f"Total number of counties reported for {STATE}: {num_counties}")
print(f"Processing complete for {STATE}")
