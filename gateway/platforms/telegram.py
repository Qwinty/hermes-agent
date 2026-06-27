"""Compatibility alias for the Telegram platform adapter.

The Telegram adapter lives in ``plugins.platforms.telegram.adapter`` as part of
Hermes' plugin-first platform layout.  Some tests and downstream integrations
still import the historical ``gateway.platforms.telegram`` module path.  Alias
this module object to the real adapter module so monkeypatching module globals
through either path affects the same code.
"""

import sys

from plugins.platforms.telegram import adapter as _adapter

sys.modules[__name__] = _adapter
