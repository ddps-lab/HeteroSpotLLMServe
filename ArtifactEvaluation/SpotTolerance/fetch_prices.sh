#!/bin/bash
set -e

# ========== AWS Credentials ==========
# Fill in either via the standard AWS chain (IAM instance profile, ~/.aws/credentials,
# `aws configure`, AWS_PROFILE) or by exporting the variables below before running.
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-PLACEHOLDER}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-PLACEHOLDER}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"

# ========== Config ==========
# Save directly into ReferenceData so re-runs overwrite the tracked snapshot
# that figure notebooks load from.
RESULT_DIR="$(dirname "$0")/../ReferenceData/SpotTolerance"
INSTANCE_TYPES=("g5.12xlarge" "g6.12xlarge" "g6e.xlarge")

# Scenario A: 2026-03-18 12:55~13:45 UTC
SCENARIO_A_START="2026-03-18T12:55:00Z"
SCENARIO_A_END="2026-03-18T13:45:00Z"

mkdir -p "$RESULT_DIR"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# ========== Fetch Spot Prices ==========
for scenario in A; do
    eval "START=\$SCENARIO_${scenario}_START"
    eval "END=\$SCENARIO_${scenario}_END"
    echo "=== Fetching Spot Prices for Scenario $scenario ($START ~ $END) ==="

    for itype in "${INSTANCE_TYPES[@]}"; do
        echo "  Instance: $itype"
        aws ec2 describe-spot-price-history \
            --instance-types "$itype" \
            --product-descriptions "Linux/UNIX" \
            --start-time "$START" \
            --end-time "$END" \
            --query 'SpotPriceHistory[*].{AZ:AvailabilityZone,Instance:InstanceType,Price:SpotPrice,Time:Timestamp}' \
            --output json > "$TMPDIR/spot_${itype//\./_}_${scenario}.json"
    done
done

# ========== Fetch On-Demand Prices ==========
echo ""
echo "=== Fetching On-Demand Prices ==="
for itype in "${INSTANCE_TYPES[@]}"; do
    echo "  Instance: $itype"
    aws pricing get-products \
        --service-code AmazonEC2 \
        --region us-east-1 \
        --filters \
            "Type=TERM_MATCH,Field=instanceType,Value=$itype" \
            "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
            "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
            "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
            "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
            "Type=TERM_MATCH,Field=location,Value=US East (N. Virginia)" \
        --query 'PriceList' \
        --output json > "$TMPDIR/ondemand_${itype//\./_}_raw.json"
done

# ========== Merge into per-scenario files ==========
echo ""
echo "=== Merging results ==="

python3 -c "
import json, glob, os

tmpdir = '$TMPDIR'
result_dir = '$RESULT_DIR'
instance_types = ['g5.12xlarge', 'g6.12xlarge', 'g6e.xlarge']

# Parse on-demand prices
ondemand_list = []
for itype in instance_types:
    safe = itype.replace('.', '_')
    raw_path = os.path.join(tmpdir, f'ondemand_{safe}_raw.json')
    with open(raw_path) as f:
        data = json.load(f)
    for item_str in data:
        item = json.loads(item_str) if isinstance(item_str, str) else item_str
        terms = item.get('terms', {}).get('OnDemand', {})
        for term_val in terms.values():
            for dim_val in term_val.get('priceDimensions', {}).values():
                price = dim_val.get('pricePerUnit', {}).get('USD', '0')
                if float(price) > 0:
                    ondemand_list.append({
                        'Instance': itype,
                        'PricePerHour_USD': price
                    })

# Merge per scenario
for scenario in ['A']:
    spot_all = []
    for itype in instance_types:
        safe = itype.replace('.', '_')
        spot_path = os.path.join(tmpdir, f'spot_{safe}_{scenario}.json')
        with open(spot_path) as f:
            spot_all.extend(json.load(f))

    result = {
        'spot': sorted(spot_all, key=lambda x: (x['Instance'], x['AZ'], x['Time'])),
        'ondemand': ondemand_list
    }

    out_path = os.path.join(result_dir, f'prices_scenario_{scenario}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  -> {out_path} ({len(spot_all)} spot entries, {len(ondemand_list)} ondemand entries)')
"

# ========== Summary ==========
echo ""
echo "=== Price Summary ==="
python3 -c "
import json

result_dir = '$RESULT_DIR'

for scenario in ['A']:
    path = f'{result_dir}/prices_scenario_{scenario}.json'
    with open(path) as f:
        data = json.load(f)

    print(f'--- Scenario {scenario} ---')
    print('  On-Demand:')
    for od in data['ondemand']:
        print(f\"    {od['Instance']}: \${od['PricePerHour_USD']}/hr\")

    print('  Spot:')
    by_instance = {}
    for s in data['spot']:
        by_instance.setdefault(s['Instance'], []).append(float(s['Price']))
    for inst, prices in sorted(by_instance.items()):
        print(f'    {inst}: min=\${min(prices):.4f}/hr, max=\${max(prices):.4f}/hr, entries={len(prices)}')
    print()
"

echo "Done! Results saved to:"
echo "  $RESULT_DIR/prices_scenario_A.json"
