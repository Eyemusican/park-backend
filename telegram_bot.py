"""
Telegram Bot for Smart Parking System

Allows drivers to:
1. View parking areas and availability
2. Get live feed links
3. Subscribe to availability notifications
"""

import os
import logging
import threading
import time
import asyncio
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logger = logging.getLogger(__name__)


class AvailabilityNotifier:
    """Background task that monitors availability and sends notifications."""

    def __init__(self, bot: 'ParkingTelegramBot', check_interval: int = 5):
        """
        Args:
            bot: The Telegram bot instance
            check_interval: Seconds between availability checks (default: 5s for responsive updates)
        """
        self.bot = bot
        self.check_interval = check_interval
        self._running = False
        self._thread = None
        self._last_availability = {}  # parking_id -> available_slots

    def start(self):
        """Start the notification worker."""
        if self._running:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Availability notifier started (checking every {self.check_interval}s)")

    def stop(self):
        """Stop the notification worker."""
        self._running = False
        logger.info("Availability notifier stopped")

    def _run(self):
        """Main loop for checking availability."""
        self._running = True
        print(f" * Availability notifier running (checking every {self.check_interval}s)")

        while self._running:
            try:
                self._check_and_notify()
            except Exception as e:
                logger.error(f"Error in availability check: {e}")
                print(f" * Notifier error: {e}")

            time.sleep(self.check_interval)

    def _check_and_notify(self):
        """Check availability and send notifications for any changes."""
        areas = self.bot._get_parking_areas()

        for area in areas:
            area_id = area['id']
            current_available = area['available_slots']
            previous_available = self._last_availability.get(area_id)

            # Skip first check (initialization)
            if previous_available is None:
                self._last_availability[area_id] = current_available
                continue

            # Notify on ANY change in availability
            if current_available != previous_available:
                self._send_state_change_notification(area, previous_available, current_available)

            self._last_availability[area_id] = current_available

    def _send_state_change_notification(self, area: dict, previous: int, current: int):
        """Send notifications to subscribed users about state changes."""
        print(f" * [NOTIFY] State change detected: {area['name']} ({previous} -> {current})", flush=True)

        subscribed_chats = self._get_subscribed_chats(area['id'])
        print(f" * [NOTIFY] Subscribers for area {area['id']}: {subscribed_chats}", flush=True)

        if not subscribed_chats:
            print(f" * [NOTIFY] No subscribers, skipping", flush=True)
            return

        # Determine if slots increased or decreased
        change = current - previous
        if change > 0:
            emoji = "🟢"
            direction = f"+{change} slot{'s' if change > 1 else ''} available"
            action = "A car left!"
        else:
            emoji = "🔴"
            direction = f"{change} slot{'s' if abs(change) > 1 else ''}"
            action = "A car arrived!"

        total = area['total_slots']
        message = (
            f"{emoji} *Parking Update*\n\n"
            f"🅿️ *{area['name']}*\n"
            f"{action}\n\n"
            f"📊 {direction}\n"
            f"Available: *{current}/{total}* slots"
        )

        # Send all messages using a single event loop
        async def send_all():
            for chat_id in subscribed_chats:
                try:
                    print(f" * [NOTIFY] Sending to {chat_id}...", flush=True)
                    await self._send_message(chat_id, message)
                    self._update_last_notified(chat_id, area['id'])
                    print(f" * [NOTIFY] Sent successfully to {chat_id}", flush=True)
                except Exception as e:
                    print(f" * [NOTIFY] FAILED to send to {chat_id}: {e}", flush=True)
                    logger.error(f"Failed to send notification to {chat_id}: {e}")

        try:
            asyncio.run(send_all())
        except RuntimeError:
            # Event loop already running, use get_event_loop
            loop = asyncio.get_event_loop()
            loop.run_until_complete(send_all())

    async def _send_message(self, chat_id: int, text: str):
        """Send a message to a chat."""
        if self.bot.app:
            await self.bot.app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="Markdown"
            )

    def _get_subscribed_chats(self, parking_id: int) -> list:
        """Get all chat IDs subscribed to a parking area."""
        try:
            conn = self.bot.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT chat_id FROM telegram_subscriptions
                WHERE parking_id = %s AND is_active = TRUE
            """, (parking_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"Error getting subscribed chats: {e}")
            return []

    def _update_last_notified(self, chat_id: int, parking_id: int):
        """Update the last notification timestamp."""
        try:
            conn = self.bot.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE telegram_subscriptions
                SET last_notified_at = CURRENT_TIMESTAMP
                WHERE chat_id = %s AND parking_id = %s
            """, (chat_id, parking_id))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating last_notified: {e}")


class ParkingTelegramBot:
    """Telegram bot for parking system"""

    def __init__(self, token: str, db_helper, api_base_url: str):
        """
        Initialize the bot.

        Args:
            token: Telegram bot token from @BotFather
            db_helper: ParkingDB instance for database access
            api_base_url: Base URL for API (e.g., 'http://localhost:5001/api')
        """
        self.token = token
        self.db = db_helper
        self.api_base_url = api_base_url
        self.app = None
        self._running = False
        self._thread = None
        self.notifier = None

    def start(self):
        """Start the bot in a background thread."""
        if self._running:
            logger.warning("Bot is already running")
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Start the availability notifier (after a short delay to let bot initialize)
        def start_notifier():
            time.sleep(2)  # Wait for bot to initialize
            self.notifier = AvailabilityNotifier(self)
            self.notifier.start()

        threading.Thread(target=start_notifier, daemon=True).start()
        logger.info("Telegram bot started in background thread")

    def stop(self):
        """Stop the bot."""
        if self.notifier:
            self.notifier.stop()
        if self.app and self._running:
            self._running = False
            logger.info("Telegram bot stopped")

    def _run(self):
        """Run the bot (called in background thread)."""
        asyncio.run(self._async_run())

    async def _async_run(self):
        """Async bot runner."""
        self.app = Application.builder().token(self.token).build()

        # Register handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("parking", self._cmd_parking))
        self.app.add_handler(CommandHandler("subscriptions", self._cmd_subscriptions))
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))

        self._running = True
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()

        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)

        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command - welcome message with main menu."""
        keyboard = [
            [InlineKeyboardButton("🅿️ View Parking Areas", callback_data="list_areas")],
            [InlineKeyboardButton("🔔 My Subscriptions", callback_data="my_subs")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "🚗 *Smart Parking Bot*\n\n"
            "I can help you find available parking and notify you when spots open up.\n\n"
            "Choose an option below:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _cmd_parking(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /parking command - list all parking areas."""
        areas = self._get_parking_areas()

        if not areas:
            await update.message.reply_text("No parking areas available.")
            return

        keyboard = []
        for area in areas:
            status_emoji = "🟢" if area['available_slots'] > 0 else "🔴"
            btn_text = f"{status_emoji} {area['name']} ({area['available_slots']}/{area['total_slots']})"
            keyboard.append([InlineKeyboardButton(btn_text, callback_data=f"area_{area['id']}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🅿️ *Parking Areas*\n\nSelect an area to see details:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _cmd_subscriptions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /subscriptions command - list user's subscriptions."""
        chat_id = update.effective_chat.id
        subs = self._get_user_subscriptions(chat_id)

        if not subs:
            await update.message.reply_text(
                "You have no active subscriptions.\n\n"
                "Use /parking to view areas and subscribe to notifications."
            )
            return

        keyboard = []
        for sub in subs:
            keyboard.append([
                InlineKeyboardButton(
                    f"🔔 {sub['parking_name']}",
                    callback_data=f"area_{sub['parking_id']}"
                ),
                InlineKeyboardButton("❌", callback_data=f"unsub_{sub['parking_id']}")
            ])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🔔 *Your Subscriptions*\n\nClick ❌ to unsubscribe:",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()

        data = query.data
        chat_id = update.effective_chat.id

        if data == "list_areas":
            areas = self._get_parking_areas()
            keyboard = []
            for area in areas:
                status_emoji = "🟢" if area['available_slots'] > 0 else "🔴"
                btn_text = f"{status_emoji} {area['name']} ({area['available_slots']}/{area['total_slots']})"
                keyboard.append([InlineKeyboardButton(btn_text, callback_data=f"area_{area['id']}")])
            keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="main_menu")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🅿️ *Parking Areas*\n\nSelect an area:",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif data == "my_subs":
            subs = self._get_user_subscriptions(chat_id)
            if not subs:
                keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="main_menu")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "You have no active subscriptions.\n\nGo to Parking Areas to subscribe.",
                    reply_markup=reply_markup
                )
            else:
                keyboard = []
                for sub in subs:
                    keyboard.append([
                        InlineKeyboardButton(f"🔔 {sub['parking_name']}", callback_data=f"area_{sub['parking_id']}"),
                        InlineKeyboardButton("❌", callback_data=f"unsub_{sub['parking_id']}")
                    ])
                keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="main_menu")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "🔔 *Your Subscriptions*\n\nClick ❌ to unsubscribe:",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )

        elif data == "main_menu":
            keyboard = [
                [InlineKeyboardButton("🅿️ View Parking Areas", callback_data="list_areas")],
                [InlineKeyboardButton("🔔 My Subscriptions", callback_data="my_subs")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                "🚗 *Smart Parking Bot*\n\nChoose an option:",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

        elif data.startswith("area_"):
            area_id = int(data.split("_")[1])
            await self._show_area_details(query, chat_id, area_id)

        elif data.startswith("sub_"):
            area_id = int(data.split("_")[1])
            self._subscribe_user(chat_id, area_id)
            await query.answer("✅ Subscribed! You'll be notified when spots open up.", show_alert=True)
            await self._show_area_details(query, chat_id, area_id)

        elif data.startswith("unsub_"):
            area_id = int(data.split("_")[1])
            self._unsubscribe_user(chat_id, area_id)
            await query.answer("🔕 Unsubscribed", show_alert=True)
            # Refresh the subscriptions view
            subs = self._get_user_subscriptions(chat_id)
            if not subs:
                keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="main_menu")]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "You have no active subscriptions.",
                    reply_markup=reply_markup
                )
            else:
                keyboard = []
                for sub in subs:
                    keyboard.append([
                        InlineKeyboardButton(f"🔔 {sub['parking_name']}", callback_data=f"area_{sub['parking_id']}"),
                        InlineKeyboardButton("❌", callback_data=f"unsub_{sub['parking_id']}")
                    ])
                keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="main_menu")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                await query.edit_message_text(
                    "🔔 *Your Subscriptions*\n\nClick ❌ to unsubscribe:",
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )

    async def _show_area_details(self, query, chat_id: int, area_id: int):
        """Show details for a specific parking area."""
        area = self._get_parking_area(area_id)
        if not area:
            await query.edit_message_text("Parking area not found.")
            return

        is_subscribed = self._is_subscribed(chat_id, area_id)

        # Build status text
        status_emoji = "🟢" if area['available_slots'] > 0 else "🔴"
        occupancy = f"{area['occupied_slots']}/{area['total_slots']}"
        rate = f"{area['occupancy_rate']:.0f}%"

        feed_url = f"{self.api_base_url}/parking-areas/{area_id}/detection/feed"

        text = (
            f"🅿️ *{area['name']}*\n\n"
            f"{status_emoji} Available: *{area['available_slots']}* slots\n"
            f"📊 Occupancy: {occupancy} ({rate})\n\n"
            f"📹 [Open Live Feed]({feed_url})\n\n"
            f"_Tap the link above to view the parking camera_"
        )

        keyboard = []
        if is_subscribed:
            keyboard.append([InlineKeyboardButton("🔕 Unsubscribe", callback_data=f"unsub_{area_id}")])
        else:
            keyboard.append([InlineKeyboardButton("🔔 Notify me when spots available", callback_data=f"sub_{area_id}")])
        keyboard.append([InlineKeyboardButton("🔄 Refresh", callback_data=f"area_{area_id}")])
        keyboard.append([InlineKeyboardButton("⬅️ Back to Areas", callback_data="list_areas")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")

    # Database helper methods
    def _get_parking_areas(self) -> list:
        """Get all parking areas with stats by calling the API."""
        try:
            response = requests.get(f"{self.api_base_url}/parking-areas", timeout=5)
            if response.status_code == 200:
                areas = response.json()
                # Normalize field names and ensure numeric types
                return [{
                    'id': int(a.get('id', 0)),
                    'name': a.get('name', ''),
                    'total_slots': int(a.get('total_slots', 0)),
                    'occupied_slots': int(a.get('occupied_slots', 0)),
                    'available_slots': int(a.get('available_slots', 0)),
                    'occupancy_rate': float(a.get('occupancy_rate', 0))
                } for a in areas]
            return []
        except Exception as e:
            logger.error(f"Error getting parking areas: {e}")
            return []

    def _get_parking_area(self, area_id: int) -> dict:
        """Get a single parking area with stats."""
        areas = self._get_parking_areas()
        for area in areas:
            if area['id'] == area_id:
                return area
        return None

    def _get_user_subscriptions(self, chat_id: int) -> list:
        """Get subscriptions for a user."""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT ts.parking_id, pa.parking_name
                FROM telegram_subscriptions ts
                JOIN parking_area pa ON ts.parking_id = pa.parking_id
                WHERE ts.chat_id = %s AND ts.is_active = TRUE
                ORDER BY pa.parking_name
            """, (chat_id,))
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [{'parking_id': r[0], 'parking_name': r[1]} for r in rows]
        except Exception as e:
            logger.error(f"Error getting subscriptions: {e}")
            return []

    def _is_subscribed(self, chat_id: int, parking_id: int) -> bool:
        """Check if user is subscribed to a parking area."""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT 1 FROM telegram_subscriptions
                WHERE chat_id = %s AND parking_id = %s AND is_active = TRUE
            """, (chat_id, parking_id))
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result is not None
        except Exception as e:
            logger.error(f"Error checking subscription: {e}")
            return False

    def _subscribe_user(self, chat_id: int, parking_id: int):
        """Subscribe user to a parking area."""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO telegram_subscriptions (chat_id, parking_id, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (chat_id, parking_id)
                DO UPDATE SET is_active = TRUE, subscribed_at = CURRENT_TIMESTAMP
            """, (chat_id, parking_id))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"User {chat_id} subscribed to parking area {parking_id}")
        except Exception as e:
            logger.error(f"Error subscribing user: {e}")

    def _unsubscribe_user(self, chat_id: int, parking_id: int):
        """Unsubscribe user from a parking area."""
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE telegram_subscriptions
                SET is_active = FALSE
                WHERE chat_id = %s AND parking_id = %s
            """, (chat_id, parking_id))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"User {chat_id} unsubscribed from parking area {parking_id}")
        except Exception as e:
            logger.error(f"Error unsubscribing user: {e}")


# Singleton instance
_bot_instance = None


def get_telegram_bot():
    """Get the singleton bot instance."""
    return _bot_instance


def init_telegram_bot(token: str, db_helper, api_base_url: str) -> ParkingTelegramBot:
    """Initialize and return the bot instance."""
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = ParkingTelegramBot(token, db_helper, api_base_url)
    return _bot_instance
