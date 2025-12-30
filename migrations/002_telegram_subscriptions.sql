-- Telegram bot subscriptions table
CREATE TABLE IF NOT EXISTS telegram_subscriptions (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL,
    parking_id INTEGER NOT NULL REFERENCES parking_area(parking_id) ON DELETE CASCADE,
    subscribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    last_notified_at TIMESTAMP,
    UNIQUE(chat_id, parking_id)
);

-- Index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_telegram_subs_parking ON telegram_subscriptions(parking_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_telegram_subs_chat ON telegram_subscriptions(chat_id) WHERE is_active = TRUE;
