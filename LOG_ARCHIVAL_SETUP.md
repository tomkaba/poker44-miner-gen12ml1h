# Miner Log Archival Setup

## Overview
Automatically compress and archive miner logs daily at 01:00 UTC to reduce disk space usage from ~200MB/day to ~20MB/day.

## Script Created
- **Location**: `scripts/archive_miner_logs.sh`
- **Function**: Gzips all `miner_*.log` files and moves them to `logs_backup/` with date suffix
- **Compression Rate**: ~10x (200MB → 20MB per day)

## Setup Instructions (Run on cx1)

### 1. Make Script Executable
```bash
chmod +x /home/tk/Poker44-subnet-main/scripts/archive_miner_logs.sh
```

### 2. Create Backup Directory
```bash
mkdir -p /home/tk/Poker44-subnet-main/logs_backup
```

### 3. Setup Cron Job
Edit crontab on cx1:
```bash
crontab -e
```

Add this line to run archival at 01:00 UTC daily:
```cron
0 1 * * * /home/tk/Poker44-subnet-main/scripts/archive_miner_logs.sh >> /home/tk/Poker44-subnet-main/logs_backup/archival.log 2>&1
```

### 4. Verify Cron Entry
```bash
crontab -l | grep archive_miner_logs
```

## How It Works

**Timing**: Runs at 01:00 UTC (after miner has been running for 24 hours)

**File Naming Pattern**:
```
Before:  miner_67.log
After:   logs_backup/miner_67.log_2026-03-22.gz
```

**Output Example**:
```
logs_backup/
├── miner_6.log_2026-03-21.gz       (20 MB)
├── miner_6.log_2026-03-22.gz       (20 MB)
├── miner_67.log_2026-03-21.gz      (20 MB)
├── miner_67.log_2026-03-22.gz      (20 MB)
└── archival.log                     (execution log)
```

## Monitoring

### Check Last Run
```bash
tail -50 /home/tk/Poker44-subnet-main/logs_backup/archival.log
```

### Check Backup Size
```bash
du -sh /home/tk/Poker44-subnet-main/logs_backup/
ls -lh /home/tk/Poker44-subnet-main/logs_backup/ | tail -20
```

### Test Script Manually
```bash
cd /home/tk/Poker44-subnet-main
./scripts/archive_miner_logs.sh
```

## Storage Benefits

**Daily savings**:
- Raw logs: 150MB per day
- Compressed: 15MB per day
- **Savings: 135MB/day or ~4GB/month**

**Example 30-day retention**:
- Raw: 4.5GB/month
- Compressed: 450MB/month
- **Monthly savings: 4GB**

## Advanced Options

### Option A: Keep Only Last 7 Days (Add to Cron)
```bash
0 2 * * * find /home/tk/Poker44-subnet-main/logs_backup -name "miner_*.log_*.gz" -mtime +7 -delete
```

### Option B: Monthly Archive to External Storage (Future)
```bash
0 0 1 * * tar -czf /external/backup/miner_logs_$(date +\%Y-\%m).tar.gz /home/tk/Poker44-subnet-main/logs_backup/
```

### Option C: Monitor Disk Space (Alert if >1GB)
```bash
0 3 * * * bash -c 'size=$(du -sb /home/tk/Poker44-subnet-main/logs_backup | cut -f1); if [ $size -gt 1073741824 ]; then echo "Backup dir >1GB" | mail -s "Storage Alert" tk@example.com; fi'
```

## Troubleshooting

**Issue**: Script doesn't run at scheduled time
```bash
# Check cron is running
sudo systemctl status cron
# Check cron logs
sudo journalctl -u cron | tail -20
```

**Issue**: Premission denied
```bash
# Check ownership
ls -l /home/tk/Poker44-subnet-main/scripts/archive_miner_logs.sh
# Should show: -rwxr-xr-x (755)
```

**Issue**: Disk still full after archival
```bash
# Check what's in logs_backup
find /home/tk/Poker44-subnet-main/logs_backup -type f -printf '%s\t%p\n' | sort -rn | head -10
# Delete old archives if needed
find /home/tk/Poker44-subnet-main/logs_backup -name "*.gz" -mtime +30 -delete
```

## Verification Checklist

- [x] Script created at `scripts/archive_miner_logs.sh`
- [ ] Made executable: `chmod +x scripts/archive_miner_logs.sh`
- [ ] Backup directory created: `mkdir -p logs_backup/`
- [ ] Cron entry added: `crontab -e`
- [ ] Cron verified: `crontab -l | grep archive`
- [ ] Test run successful: `./scripts/archive_miner_logs.sh`
- [ ] Check archival.log for success messages

## Next Steps

1. Copy this script to cx1
2. Run setup commands above
3. Wait until 01:00 UTC tomorrow to verify first run
4. Check `logs_backup/archival.log` for results
