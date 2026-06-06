"""Shared test configuration: force a non-interactive matplotlib backend."""

import matplotlib

matplotlib.use("Agg")
