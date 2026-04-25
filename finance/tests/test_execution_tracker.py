from __future__ import annotations

from unittest.mock import MagicMock

from finance.execution.position_tracker import InstrumentState, PositionTracker


def _make_journal() -> MagicMock:
    return MagicMock()


class TestInstrumentState:
    def test_default_state_has_no_direction(self):
        state = InstrumentState(strategy_id="srs_fdxs")
        assert state.direction is None
        assert state.long_order_id is None
        assert state.short_order_id is None
        assert state.fill_price is None
        assert state.trail_order_id is None


class TestPositionTrackerGetAndSet:
    def test_get_returns_none_for_unknown_symbol(self):
        tracker = PositionTracker(journal=_make_journal())
        assert tracker.get("FDXS") is None

    def test_set_pending_stores_state(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="FDXS", strategy_id="srs_fdxs", long_order_id=1, short_order_id=2)
        state = tracker.get("FDXS")
        assert state is not None
        assert state.long_order_id == 1
        assert state.short_order_id == 2
        assert state.strategy_id == "srs_fdxs"
        assert state.direction is None

    def test_set_pending_registers_both_order_ids(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="asrs_mnq", long_order_id=10, short_order_id=11)
        assert tracker.order_id_to_strategy(10) == "asrs_mnq"
        assert tracker.order_id_to_strategy(11) == "asrs_mnq"


class TestOrderIdToStrategy:
    def test_returns_strategy_id_for_registered_order(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="srs_mnq", long_order_id=5, short_order_id=6)
        assert tracker.order_id_to_strategy(5) == "srs_mnq"

    def test_returns_none_for_unknown_order(self):
        tracker = PositionTracker(journal=_make_journal())
        assert tracker.order_id_to_strategy(999) is None


class TestOnFill:
    def test_on_fill_sets_direction_and_fill_price(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="FDXS", strategy_id="srs_fdxs", long_order_id=1, short_order_id=2)
        tracker.on_fill(symbol="FDXS", filled_order_id=1, direction="long", fill_price=18500.0)
        state = tracker.get("FDXS")
        assert state.direction == "long"
        assert state.fill_price == 18500.0

    def test_on_fill_sets_trail_order_id(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="FDXS", strategy_id="srs_fdxs", long_order_id=1, short_order_id=2)
        tracker.on_fill(symbol="FDXS", filled_order_id=1, direction="long", fill_price=18500.0, trail_order_id=99)
        state = tracker.get("FDXS")
        assert state.trail_order_id == 99

    def test_on_fill_journals_fill_event(self):
        journal = _make_journal()
        tracker = PositionTracker(journal=journal)
        tracker.set_pending(symbol="FDXS", strategy_id="srs_fdxs", long_order_id=1, short_order_id=2)
        tracker.on_fill(symbol="FDXS", filled_order_id=1, direction="long", fill_price=18500.0)
        journal.write.assert_called_once()
        call_entry = journal.write.call_args[0][0]
        assert call_entry["event"] == "fill"
        assert call_entry["symbol"] == "FDXS"

    def test_on_fill_registers_trail_order_id_for_routing(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="FDXS", strategy_id="srs_fdxs", long_order_id=1, short_order_id=2)
        tracker.on_fill(symbol="FDXS", filled_order_id=1, direction="long", fill_price=18500.0, trail_order_id=99)
        assert tracker.order_id_to_strategy(99) == "srs_fdxs"


class TestOnClose:
    def test_on_close_removes_state(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="srs_mnq", long_order_id=3, short_order_id=4)
        tracker.on_fill(symbol="MNQ", filled_order_id=3, direction="long", fill_price=21000.0)
        tracker.on_close(symbol="MNQ")
        assert tracker.get("MNQ") is None

    def test_on_close_removes_order_id_mappings(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="srs_mnq", long_order_id=3, short_order_id=4)
        tracker.on_fill(symbol="MNQ", filled_order_id=3, direction="long", fill_price=21000.0)
        tracker.on_close(symbol="MNQ")
        assert tracker.order_id_to_strategy(3) is None
        assert tracker.order_id_to_strategy(4) is None

    def test_on_close_removes_trail_order_id_mapping(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="srs_mnq", long_order_id=3, short_order_id=4)
        tracker.on_fill(symbol="MNQ", filled_order_id=3, direction="long", fill_price=21000.0, trail_order_id=55)
        tracker.on_close(symbol="MNQ")
        assert tracker.order_id_to_strategy(55) is None


class TestReconcile:
    def test_reconcile_rebuilds_state_from_ibkr_positions(self):
        tracker = PositionTracker(journal=_make_journal())

        mock_position = MagicMock()
        mock_position.contract.localSymbol = "FDXS"
        mock_position.position = 1.0

        mock_order = MagicMock()
        mock_order.orderId = 7
        mock_order.contract.localSymbol = "FDXS"
        mock_order.order.action = "BUY"
        mock_order.order.orderType = "TRAIL"

        tracker.reconcile(
            ibkr_positions=[mock_position],
            ibkr_orders=[mock_order],
            strategy_id_for_symbol={"FDXS": "srs_fdxs"},
        )

        state = tracker.get("FDXS")
        assert state is not None
        assert state.strategy_id == "srs_fdxs"
        assert state.direction == "long"
        assert state.trail_order_id == 7

    def test_reconcile_registers_trail_order_id_for_routing(self):
        tracker = PositionTracker(journal=_make_journal())

        mock_position = MagicMock()
        mock_position.contract.localSymbol = "FDXS"
        mock_position.position = 1.0

        mock_order = MagicMock()
        mock_order.orderId = 7
        mock_order.contract.localSymbol = "FDXS"
        mock_order.order.orderType = "TRAIL"

        tracker.reconcile(
            ibkr_positions=[mock_position],
            ibkr_orders=[mock_order],
            strategy_id_for_symbol={"FDXS": "srs_fdxs"},
        )

        assert tracker.order_id_to_strategy(7) == "srs_fdxs"

    def test_reconcile_clears_stale_state(self):
        tracker = PositionTracker(journal=_make_journal())
        tracker.set_pending(symbol="MNQ", strategy_id="srs_mnq", long_order_id=1, short_order_id=2)
        tracker.reconcile(ibkr_positions=[], ibkr_orders=[], strategy_id_for_symbol={})
        assert tracker.get("MNQ") is None
