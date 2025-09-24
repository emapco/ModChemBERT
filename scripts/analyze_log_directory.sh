#!/bin/bash
# Copyright 2025 Emmanuel Cortes, All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to analyze log files in a directory for:
# - Maximum Val mean-roc_auc_score and minimum Val rms_score (during training)
# - Final triplicate test results (mean ± std dev format)
# Usage: ./analyze_log_directory.sh <directory_path>

set -e

# Function to display usage
usage() {
    echo "Usage: $0 <directory_path>"
    echo "Example: $0 chemberta3/chemberta3_benchmarking/models_benchmarking/modchembert_benchmark/logs_modchembert_regression_checkpoint-27088"
    exit 1
}

# Function to find maximum Val mean-roc_auc_score in a file
find_max_roc_auc() {
    local file="$1"
    local result=$(grep "Val mean-roc_auc_score:" "$file" 2>/dev/null | \
                   sed 's/.*Val mean-roc_auc_score: \([0-9]*\.[0-9]*\).*/\1/' | \
                   awk 'BEGIN{max=0; count=0} {count++; if($1 > max) max=$1} END{if(count==0) print "N/A"; else printf "%.4f", max}')
    echo "$result"
}

# Function to find minimum Val rms_score in a file
find_min_rms_score() {
    local file="$1"
    local result=$(grep "Val rms_score:" "$file" 2>/dev/null | \
                   sed 's/.*Val rms_score: \([0-9]*\.[0-9]*\).*/\1/' | \
                   awk 'BEGIN{min=999999; count=0} {count++; if($1 < min) min=$1} END{if(count==0) print "N/A"; else printf "%.4f", min}')
    echo "$result"
}

# Function to extract final triplicate ROC AUC results in format "avg ± std"
find_final_roc_auc() {
    local file="$1"
    local result=$(grep "Final Triplicate Test Results.*mean-roc_auc_score:" "$file" 2>/dev/null | \
                   sed 's/.*Avg mean-roc_auc_score: \([0-9]*\.[0-9]*\), Std Dev: \([0-9]*\.[0-9]*\).*/\1 ± \2/' | \
                   head -1)
    if [ -z "$result" ]; then
        echo "N/A"
    else
        echo "$result"
    fi
}

# Function to extract final triplicate RMS score results in format "avg ± std"
find_final_rms_score() {
    local file="$1"
    local result=$(grep "Final Triplicate Test Results.*rms_score:" "$file" 2>/dev/null | \
                   sed 's/.*Avg rms_score: \([0-9]*\.[0-9]*\), Std Dev: \([0-9]*\.[0-9]*\).*/\1 ± \2/' | \
                   head -1)
    if [ -z "$result" ]; then
        echo "N/A"
    else
        echo "$result"
    fi
}

# Function to get basename without extension for cleaner display
get_clean_filename() {
    local filepath="$1"
    basename "$filepath" .log
}

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory specified"
    usage
fi

DIRECTORY="$1"

# Check if directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' not found"
    exit 1
fi

# Check if directory contains any .log files
if ! find "$DIRECTORY" -maxdepth 1 -name "*.log" -type f | grep -q .; then
    echo "No .log files found in directory: $DIRECTORY"
    exit 1
fi

echo "Analyzing log files in directory: $DIRECTORY"
echo "=================================================="
echo

# Create table header
printf "File \t| Max Validation ROC AUC \t| Min Validation RMS Score \t| Final Test ROC AUC \t| Final Test RMS Score\n"
echo "------------------------------------------------------------------------------"
# Initialize accumulators for averages (ignore N/A)
sum_max_roc=0
count_max_roc=0
sum_min_rms=0
count_min_rms=0
sum_final_roc=0
count_final_roc=0
sum_final_rms=0
count_final_rms=0

# Also track averages excluding outliers (> 10)
sum_max_roc_no_outlier=0
count_max_roc_no_outlier=0
sum_min_rms_no_outlier=0
count_min_rms_no_outlier=0
sum_final_roc_no_outlier=0
count_final_roc_no_outlier=0
sum_final_rms_no_outlier=0
count_final_rms_no_outlier=0

# Helper to add floating numbers using awk
fadd() {
    awk -v a="$1" -v b="$2" 'BEGIN { if(a=="") a=0; if(b=="") b=0; printf "%.10f", a + b }'
}

# Trim function for extracting numeric means from "avg ± std"
trim() {
    local s="$1"
    # shellcheck disable=SC2001
    echo "$s" | sed 's/^\s*//;s/\s*$//'
}

# Float comparison helper: returns success (0) if x <= 10, else failure
leq_10() {
    awk -v x="$1" 'BEGIN { if ((x+0) <= 10) exit 0; else exit 1 }'
}

# Process each .log file in the directory
while IFS= read -r -d '' file; do
    clean_name=$(get_clean_filename "$file")
    max_roc=$(find_max_roc_auc "$file")
    min_rms=$(find_min_rms_score "$file")
    final_roc=$(find_final_roc_auc "$file")
    final_rms=$(find_final_rms_score "$file")
    printf "%s\t\t%s\t\t%s\t\t%s\t\t%s\n" "$clean_name" "$max_roc" "$min_rms" "$final_roc" "$final_rms"

    # Accumulate for averages where values are numeric (ignore N/A)
    if [[ "$max_roc" != "N/A" ]]; then
        sum_max_roc=$(fadd "$sum_max_roc" "$max_roc")
        count_max_roc=$((count_max_roc + 1))
        if leq_10 "$max_roc"; then
            sum_max_roc_no_outlier=$(fadd "$sum_max_roc_no_outlier" "$max_roc")
            count_max_roc_no_outlier=$((count_max_roc_no_outlier + 1))
        fi
    fi
    if [[ "$min_rms" != "N/A" ]]; then
        sum_min_rms=$(fadd "$sum_min_rms" "$min_rms")
        count_min_rms=$((count_min_rms + 1))
        if leq_10 "$min_rms"; then
            sum_min_rms_no_outlier=$(fadd "$sum_min_rms_no_outlier" "$min_rms")
            count_min_rms_no_outlier=$((count_min_rms_no_outlier + 1))
        fi
    fi
    if [[ "$final_roc" != "N/A" ]]; then
        # extract mean component before the ± symbol
        mean_part=$(trim "${final_roc%%±*}")
        if [[ "$mean_part" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            sum_final_roc=$(fadd "$sum_final_roc" "$mean_part")
            count_final_roc=$((count_final_roc + 1))
            if leq_10 "$mean_part"; then
                sum_final_roc_no_outlier=$(fadd "$sum_final_roc_no_outlier" "$mean_part")
                count_final_roc_no_outlier=$((count_final_roc_no_outlier + 1))
            fi
        fi
    fi
    if [[ "$final_rms" != "N/A" ]]; then
        mean_part=$(trim "${final_rms%%±*}")
        if [[ "$mean_part" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            sum_final_rms=$(fadd "$sum_final_rms" "$mean_part")
            count_final_rms=$((count_final_rms + 1))
            if leq_10 "$mean_part"; then
                sum_final_rms_no_outlier=$(fadd "$sum_final_rms_no_outlier" "$mean_part")
                count_final_rms_no_outlier=$((count_final_rms_no_outlier + 1))
            fi
        fi
    fi
done < <(find "$DIRECTORY" -maxdepth 1 -name "*.log" -type f -print0 | sort -z)

# Compute and print averages row
avg_or_na() {
    local sum="$1"
    local cnt="$2"
    if [[ "$cnt" -gt 0 ]]; then
        awk -v s="$sum" -v c="$cnt" 'BEGIN { printf "%.4f", s/c }'
    else
        echo "N/A"
    fi
}

avg_max_roc=$(avg_or_na "$sum_max_roc" "$count_max_roc")
avg_min_rms=$(avg_or_na "$sum_min_rms" "$count_min_rms")
avg_final_roc=$(avg_or_na "$sum_final_roc" "$count_final_roc")
avg_final_rms=$(avg_or_na "$sum_final_rms" "$count_final_rms")

# Averages excluding outliers (> 10)
avg_max_roc_no_outlier=$(avg_or_na "$sum_max_roc_no_outlier" "$count_max_roc_no_outlier")
avg_min_rms_no_outlier=$(avg_or_na "$sum_min_rms_no_outlier" "$count_min_rms_no_outlier")
avg_final_roc_no_outlier=$(avg_or_na "$sum_final_roc_no_outlier" "$count_final_roc_no_outlier")
avg_final_rms_no_outlier=$(avg_or_na "$sum_final_rms_no_outlier" "$count_final_rms_no_outlier")

echo "------------------------------------------------------------------------------"
printf "AVERAGE\t\t%s / %s\t\t%s / %s\t\t%s / %s\t\t%s / %s\n" \
    "$avg_max_roc_no_outlier" "$avg_max_roc" \
    "$avg_min_rms_no_outlier" "$avg_min_rms" \
    "$avg_final_roc_no_outlier" "$avg_final_roc" \
    "$avg_final_rms_no_outlier" "$avg_final_rms"
