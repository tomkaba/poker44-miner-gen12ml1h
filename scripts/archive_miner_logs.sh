#!/bin/bash
# Archive miner logs daily to logs_backup/ with timestamp
# Run via cron: 0 1 * * * /home/tk/Poker44-subnet-main/scripts/archive_miner_logs.sh

set -e

LOG_DIR="/home/tk/Poker44-subnet-main"
BACKUP_DIR="${LOG_DIR}/logs_backup"
DATESTAMP=$(date -u +%Y-%m-%d)  # Yesterday's date (runs at 0:01, so logs are from yesterday)
TEMP_DIR=$(mktemp -d)

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Find all miner_*.log files (not compressed)
if ls "$LOG_DIR"/miner_*.log 1> /dev/null 2>&1; then
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Starting log archival..."
    
    cd "$LOG_DIR"
    
    # Compress each log file individually
    for logfile in miner_*.log; do
        if [ -f "$logfile" ]; then
            gzip -v "$logfile"
            compressed_file="${logfile}.gz"
            
            # Move to backup directory with date
            mv "$compressed_file" "$BACKUP_DIR/${logfile}_${DATESTAMP}.gz"
            echo "✓ Archived: ${logfile}_${DATESTAMP}.gz"
        fi
    done
    
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Archival complete."
    
    # Cleanup temp dir
    rm -rf "$TEMP_DIR"
    
    # Log rotation stats
    size=$(du -sh "$BACKUP_DIR" | cut -f1)
    count=$(ls "$BACKUP_DIR"/*.gz 2>/dev/null | wc -l)
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] Backup directory: $count files, total size: $size"
else
    echo "[$(date -u +'%Y-%m-%d %H:%M:%S UTC')] No log files found to archive."
fi
