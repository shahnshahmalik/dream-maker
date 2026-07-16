# Deployment — Ubuntu Server

## Install the systemd service

```bash
# Copy unit file
sudo cp deploy/sniper.service /etc/systemd/system/sniper.service

# Reload systemd and enable
sudo systemctl daemon-reload
sudo systemctl enable sniper
sudo systemctl start sniper
```

## Monitor

```bash
# Live logs (guardian + sniper combined)
journalctl -u sniper -f

# Sniper stdout log (session-level detail)
tail -f /home/ubuntu/projects/dream-maker/logs/sniper.log

# Guardian log
tail -f /home/ubuntu/projects/dream-maker/logs/sniper_guardian.log

# Service status
systemctl status sniper
```

## Stop / Restart

```bash
sudo systemctl stop sniper       # graceful stop (SIGTERM → child exits)
sudo systemctl restart sniper    # restart guardian + child
```

## Override configuration without editing the unit file

```bash
sudo systemctl edit sniper
# Add an [Service] block, e.g.:
# [Service]
# Environment=FORCE_COLOR=1
```

## How it works

```
systemd
  └── sniper_guardian.py   (always running, crash recovery, Telegram alerts)
        └── scripts/sniper.py   (24x7 outer loop)
              ├── market closed  → sleeps using utils/market_hours
              ├── Tuesday open   → Phase 1 (ExpirySniperConfig) + Phase 2 (ThetaKillConfig)
              └── other days     → SniperEngine(NonExpirySniperConfig)
```

The guardian monitors `state/sniper_heartbeat.json`. If the heartbeat is stale
for more than 5 minutes, it kills and restarts the child. If there are more than
5 crashes in 10 minutes, it halts and sends a Telegram alert.
