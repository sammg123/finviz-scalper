"""
Google Sheets Trade Logger
==========================
Logs every paper trade to a Google Sheet and maintains a live dashboard.

Setup (OAuth — one-time browser auth, no service account key needed):
  1. Go to https://console.cloud.google.com/
  2. Create a project (or use existing)
  3. Enable "Google Sheets API" and "Google Drive API"
  4. Go to Credentials -> Create Credentials -> OAuth client ID
     - Application type: Desktop app
     - Download the JSON and save as google_creds.json in this directory
  5. Create a Google Sheet and copy the Sheet ID from the URL:
     https://docs.google.com/spreadsheets/d/SHEET_ID/edit
  6. Put the sheet ID in SHEET_ID in paper_trader.py
  7. First run will open a browser to authorize — after that it's cached.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

log = logging.getLogger("SheetsLogger")

CREDS_FILE = Path("google_creds.json")
TOKEN_FILE = Path("google_token.json")


class SheetsLogger:
    """Logs trades to Google Sheets and updates a live dashboard."""

    # Dashboard layout constants
    DASH_TITLE_ROW = 1
    DASH_STATS_START = 3
    DASH_COIN_START = 13
    DASH_RECENT_START = 22

    def __init__(self, sheet_id: str, creds_file: Path = CREDS_FILE):
        self.sheet_id = sheet_id
        self.gc = None
        self.spreadsheet = None
        self.trades_ws = None
        self.dash_ws = None
        self._connected = False

        try:
            import gspread
            # Try service account first (if google_creds.json is a service account key)
            if creds_file.exists():
                try:
                    self.gc = gspread.service_account(filename=str(creds_file))
                    self.spreadsheet = self.gc.open_by_key(sheet_id)
                    self._setup_sheets()
                    self._connected = True
                    log.info(f"Google Sheets connected (service account): {self.spreadsheet.title}")
                    return
                except Exception:
                    pass  # Not a service account key, try OAuth

            # OAuth flow: uses google_creds.json as OAuth client ID
            # First run opens browser, then caches token in google_token.json
            if creds_file.exists():
                self.gc = gspread.oauth(
                    credentials_filename=str(creds_file),
                    authorized_user_filename=str(TOKEN_FILE),
                )
            else:
                # Fall back to default gspread OAuth (~/.config/gspread/)
                self.gc = gspread.oauth(
                    authorized_user_filename=str(TOKEN_FILE),
                )
            self.spreadsheet = self.gc.open_by_key(sheet_id)
            self._setup_sheets()
            self._connected = True
            log.info(f"Google Sheets connected (OAuth): {self.spreadsheet.title}")
        except Exception as e:
            log.error(f"Google Sheets connection failed: {e}")
            log.error("See sheets_logger.py header for setup instructions.")

    @property
    def connected(self) -> bool:
        return self._connected

    def _setup_sheets(self):
        """Ensure Trades and Dashboard sheets exist with headers."""
        existing = [ws.title for ws in self.spreadsheet.worksheets()]

        # --- Trades sheet ---
        if "Trades" not in existing:
            self.trades_ws = self.spreadsheet.add_worksheet("Trades", rows=1000, cols=13)
        else:
            self.trades_ws = self.spreadsheet.worksheet("Trades")

        # Check if headers exist
        row1 = self.trades_ws.row_values(1)
        if not row1 or row1[0] != "Time":
            headers = [
                "Time", "Coin", "Side", "Entry", "Exit",
                "Raw P&L %", "Fees %", "Net P&L %", "P&L $",
                "Equity", "Reason", "Bars", "Result",
            ]
            self.trades_ws.update("A1:M1", [headers])
            self.trades_ws.format("A1:M1", {
                "textFormat": {"bold": True},
                "backgroundColor": {"red": 0.15, "green": 0.15, "blue": 0.15},
                "horizontalAlignment": "CENTER",
            })
            self.trades_ws.format("A1:M1", {
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
            })

        # --- Dashboard sheet ---
        if "Dashboard" not in existing:
            self.dash_ws = self.spreadsheet.add_worksheet("Dashboard", rows=40, cols=8)
        else:
            self.dash_ws = self.spreadsheet.worksheet("Dashboard")

        # Move Dashboard to first position
        self.spreadsheet.reorder_worksheets(
            [self.dash_ws] + [ws for ws in self.spreadsheet.worksheets() if ws.title != "Dashboard"]
        )

        # Remove default Sheet1 if it exists and is empty
        for ws in self.spreadsheet.worksheets():
            if ws.title == "Sheet1" and len(self.spreadsheet.worksheets()) > 1:
                try:
                    self.spreadsheet.del_worksheet(ws)
                except Exception:
                    pass
                break

    def log_trade(self, trade: dict):
        """Append a trade row to the Trades sheet."""
        if not self._connected:
            return

        try:
            result = "WIN" if trade.get("pnl_dollar", "$0").startswith("$+") or \
                     (not trade.get("pnl_dollar", "").startswith("$-") and
                      trade.get("pnl_dollar", "$0") != "$+0.00" and
                      float(trade.get("pnl_dollar", "$0").replace("$", "").replace(",", "").replace("+", "")) > 0) \
                     else "LOSS"

            row = [
                trade.get("time", ""),
                trade.get("coin", ""),
                trade.get("side", "").upper(),
                trade.get("entry", ""),
                trade.get("exit", ""),
                trade.get("raw_pnl_pct", ""),
                trade.get("fees_pct", ""),
                trade.get("net_pnl_pct", ""),
                trade.get("pnl_dollar", ""),
                trade.get("equity_after", ""),
                trade.get("reason", ""),
                trade.get("bars_held", ""),
                result,
            ]
            self.trades_ws.append_row(row, value_input_option="USER_ENTERED")
            log.info(f"  [SHEETS] Trade logged: {trade.get('coin')} {trade.get('side')} -> {result}")
        except Exception as e:
            log.error(f"  [SHEETS] Failed to log trade: {e}")

    def update_dashboard(self, stats: dict):
        """Push full dashboard stats to the Dashboard sheet."""
        if not self._connected:
            return

        try:
            cells = []

            # Title
            cells.append({"range": "A1:H1", "values": [[
                "PAPER TRADER DASHBOARD", "", "", "", "", "", "",
                f"Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
            ]]})

            # Overall stats block
            r = self.DASH_STATS_START
            equity = stats.get("equity", 0)
            peak = stats.get("peak_equity", 0)
            starting = stats.get("starting_equity", 10000)
            dd = (equity - peak) / peak if peak > 0 else 0
            total_ret = (equity - starting) / starting if starting > 0 else 0
            total_trades = stats.get("total_trades", 0)
            winning = stats.get("winning_trades", 0)
            win_rate = winning / total_trades if total_trades > 0 else 0
            total_pnl = stats.get("total_pnl", 0)
            avg_win = stats.get("avg_win", 0)
            avg_loss = stats.get("avg_loss", 0)
            profit_factor = stats.get("profit_factor", 0)
            max_consec_loss = stats.get("max_consec_loss", 0)

            overall = [
                ["OVERALL PERFORMANCE", "", "", "", "RISK METRICS", "", "", ""],
                ["Equity", f"${equity:,.2f}", "", "", "Peak Equity", f"${peak:,.2f}", "", ""],
                ["Total Return", f"{total_ret:+.2%}", "", "", "Max Drawdown", f"{dd:.2%}", "", ""],
                ["Net P&L", f"${total_pnl:+,.2f}", "", "", "Profit Factor", f"{profit_factor:.2f}" if profit_factor else "N/A", "", ""],
                ["Total Trades", str(total_trades), "", "", "Max Consec Losses", str(max_consec_loss), "", ""],
                ["Win Rate", f"{win_rate:.1%}", "", "", "Avg Win", f"${avg_win:+,.2f}" if avg_win else "$0.00", "", ""],
                ["Winning", str(winning), "", "", "Avg Loss", f"${avg_loss:+,.2f}" if avg_loss else "$0.00", "", ""],
                ["Losing", str(total_trades - winning), "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", ""],
            ]
            cells.append({"range": f"A{r}:H{r + len(overall) - 1}", "values": overall})

            # Per-coin breakdown
            r = self.DASH_COIN_START
            coin_stats = stats.get("coin_stats", {})
            coin_header = [["PER-COIN BREAKDOWN", "", "", "", "", "", "", ""]]
            coin_cols = [["Coin", "Trades", "Win Rate", "Net P&L", "Avg P&L", "Sizing", "Cooldown", ""]]
            coin_rows = []
            for coin, cs in sorted(coin_stats.items()):
                coin_rows.append([
                    coin,
                    str(cs.get("trades", 0)),
                    f"{cs.get('win_rate', 0):.0%}",
                    f"${cs.get('total_pnl', 0):+,.2f}",
                    f"${cs.get('avg_pnl', 0):+,.2f}",
                    f"{cs.get('momentum_mult', 1):.2f}x",
                    f"{cs.get('cooldown', 0)} bars" if cs.get('cooldown', 0) > 0 else "-",
                    "",
                ])
            if not coin_rows:
                coin_rows.append(["No trades yet", "", "", "", "", "", "", ""])

            all_coin = coin_header + coin_cols + coin_rows
            cells.append({"range": f"A{r}:H{r + len(all_coin) - 1}", "values": all_coin})

            # Open positions
            r = self.DASH_RECENT_START
            positions = stats.get("open_positions", [])
            pos_header = [["OPEN POSITIONS", "", "", "", "", "", "", ""]]
            pos_cols = [["Coin", "Side", "Entry", "Current", "Unrealized", "Stop", "Bars", ""]]
            pos_rows = []
            for p in positions:
                pos_rows.append([
                    p.get("coin", ""),
                    p.get("side", "").upper(),
                    f"${p.get('entry', 0):,.4f}",
                    f"${p.get('current', 0):,.4f}",
                    f"${p.get('unrealized', 0):+,.2f}",
                    f"${p.get('stop', 0):,.4f}",
                    str(p.get("bars", 0)),
                    "",
                ])
            if not pos_rows:
                pos_rows.append(["No open positions", "", "", "", "", "", "", ""])

            all_pos = pos_header + pos_cols + pos_rows
            cells.append({"range": f"A{r}:H{r + len(all_pos) - 1}", "values": all_pos})

            # Batch update
            self.dash_ws.batch_update(cells, value_input_option="USER_ENTERED")

            # Format headers
            self._format_dashboard()

            log.info(f"  [SHEETS] Dashboard updated")
        except Exception as e:
            log.error(f"  [SHEETS] Failed to update dashboard: {e}")

    def _format_dashboard(self):
        """Apply formatting to dashboard headers."""
        try:
            bold = {"textFormat": {"bold": True}}
            title_fmt = {
                "textFormat": {"bold": True, "fontSize": 14},
            }
            section_fmt = {
                "textFormat": {"bold": True, "fontSize": 11},
                "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.95},
            }

            self.dash_ws.format("A1", title_fmt)
            self.dash_ws.format(f"A{self.DASH_STATS_START}", section_fmt)
            self.dash_ws.format(f"E{self.DASH_STATS_START}", section_fmt)
            self.dash_ws.format(f"A{self.DASH_COIN_START}", section_fmt)
            col_row = self.DASH_COIN_START + 1
            self.dash_ws.format(f"A{col_row}:G{col_row}", bold)
            self.dash_ws.format(f"A{self.DASH_RECENT_START}", section_fmt)
            pos_col_row = self.DASH_RECENT_START + 1
            self.dash_ws.format(f"A{pos_col_row}:G{pos_col_row}", bold)
        except Exception:
            pass  # Formatting is cosmetic, don't crash on failure
