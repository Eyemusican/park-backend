# Telegram Bot Setup Guide

## Prerequisites

- Park-backend running on port 5001
- PostgreSQL database configured

## Step 1: Create a Telegram Bot

1. Open Telegram and search for `@BotFather`
2. Start a chat and send `/newbot`
3. Follow the prompts:
   - Enter a name for your bot (e.g., "Smart Parking Bot")
   - Enter a username (must end in `bot`, e.g., `SmartParkingBot`)
4. BotFather will give you an API token like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`
5. Save this token securely

## Step 2: Configure the Backend

1. Copy the `.env.example` to `.env` if not already done:
   ```bash
   cp .env.example .env
   ```

2. Add your bot token to `.env`:
   ```bash
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

3. (Optional) If your backend is accessible externally, set the API URL:
   ```bash
   TELEGRAM_API_BASE_URL=https://your-domain.com/api
   ```

## Step 3: Apply Database Migration

```bash
docker exec -i parking-postgres psql -U parking_user -d parking_db < migrations/002_telegram_subscriptions.sql
```

## Step 4: Start the Backend

```bash
python server.py
```

You should see:
```
* Telegram bot started
```

## Step 5: Test the Bot

1. Open Telegram and search for your bot username
2. Send `/start` to begin
3. Test the commands:
   - `/parking` - View parking areas
   - `/subscriptions` - View your subscriptions

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and main menu |
| `/parking` | List all parking areas with availability |
| `/subscriptions` | View and manage your subscriptions |

## Features

### View Parking Areas
- See real-time slot availability
- Green dot = spots available, Red dot = full

### Live Feed Links
- Each parking area shows a link to the MJPEG live feed
- Open in browser to view the parking lot camera

### Availability Notifications
- Subscribe to parking areas to get notified when spots open up
- Notifications are sent when a full lot gets free spots
- Unsubscribe anytime via the bot

## Troubleshooting

### Bot not responding
- Check that `TELEGRAM_BOT_TOKEN` is set correctly
- Verify the token with: `curl https://api.telegram.org/bot<TOKEN>/getMe`

### Feed URL not working
- Ensure the backend is accessible from where you're viewing
- For external access, you may need to set up port forwarding or a reverse proxy

### Notifications not arriving
- Verify you're subscribed (check via `/subscriptions`)
- Check server logs for notification errors
- Ensure the parking area was fully occupied before spots opened
