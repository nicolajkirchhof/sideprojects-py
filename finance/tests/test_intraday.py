import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timezone

from finance.utils import intraday


@pytest.fixture()
def data_dir(tmp_path):
  """Create a temporary intraday data directory with one index symbol."""
  idx_dir = tmp_path / 'index'
  idx_dir.mkdir()

  times = pd.date_range('2024-01-02 09:30', periods=600, freq='min', tz='UTC')
  rng = np.random.default_rng(42)
  df = pd.DataFrame({
    'o': rng.normal(4800, 10, 600),
    'h': rng.normal(4810, 10, 600),
    'l': rng.normal(4790, 10, 600),
    'c': rng.normal(4800, 10, 600),
    'v': rng.integers(100, 5000, 600).astype(float),
    'a': rng.normal(4800, 10, 600),
    'bc': rng.integers(10, 200, 600).astype(float),
    'hvo': rng.normal(0.15, 0.01, 600),
    'hvh': rng.normal(0.16, 0.01, 600),
    'hvl': rng.normal(0.14, 0.01, 600),
    'hvc': rng.normal(0.15, 0.01, 600),
    'ivo': rng.normal(0.18, 0.01, 600),
    'ivh': rng.normal(0.19, 0.01, 600),
    'ivl': rng.normal(0.17, 0.01, 600),
    'ivc': rng.normal(0.18, 0.01, 600),
  }, index=times)
  df.index.name = 'time'
  df.to_parquet(idx_dir / 'SPX.parquet')

  # Also create an OHLCV-only CFD file
  cfd_dir = tmp_path / 'cfd'
  cfd_dir.mkdir()
  df_cfd = df[['o', 'h', 'l', 'c', 'v', 'a', 'bc']].copy()
  df_cfd.to_parquet(cfd_dir / 'IBDE40.parquet')

  return tmp_path


class TestGetBars:
  def test_returns_full_range(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    df = intraday.get_bars('SPX', 'index', start, end, data_dir=data_dir)
    assert len(df) == 600
    assert df.index.name == 'time'
    assert 'o' in df.columns and 'c' in df.columns

  def test_filters_by_date_range(self, data_dir):
    start = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc)
    df = intraday.get_bars('SPX', 'index', start, end, data_dir=data_dir)
    assert len(df) == 60
    assert df.index.min() >= pd.Timestamp(start)
    assert df.index.max() < pd.Timestamp(end)

  def test_resamples_to_5min(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    df = intraday.get_bars('SPX', 'index', start, end, period='5min', data_dir=data_dir)
    assert len(df) == 120
    assert 'o' in df.columns and 'v' in df.columns

  def test_resamples_to_1h(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    df = intraday.get_bars('SPX', 'index', start, end, period='1h', data_dir=data_dir)
    assert len(df) == 11  # 9:30-10:00(30min), then 10 full hours

  def test_resamples_ohlcv_correctly(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 9, 35, tzinfo=timezone.utc)
    raw = intraday.get_bars('SPX', 'index', start, end, data_dir=data_dir)
    resampled = intraday.get_bars('SPX', 'index', start, end, period='5min', data_dir=data_dir)
    assert len(resampled) == 1
    row = resampled.iloc[0]
    assert row['o'] == raw.iloc[0]['o']
    assert row['c'] == raw.iloc[-1]['c']
    assert row['h'] == raw['h'].max()
    assert row['l'] == raw['l'].min()
    assert row['v'] == pytest.approx(raw['v'].sum())

  def test_missing_symbol_returns_empty(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    df = intraday.get_bars('NONEXISTENT', 'index', start, end, data_dir=data_dir)
    assert df.empty

  def test_ohlcv_only_file(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    df = intraday.get_bars('IBDE40', 'cfd', start, end, data_dir=data_dir)
    assert len(df) == 600
    assert 'hvo' not in df.columns


class TestGetLast:
  def test_returns_last_timestamp(self, data_dir):
    last = intraday.get_last('SPX', 'index', data_dir=data_dir)
    assert last == pd.Timestamp('2024-01-02 19:29:00', tz='UTC')

  def test_missing_symbol_returns_none(self, data_dir):
    last = intraday.get_last('NONEXISTENT', 'index', data_dir=data_dir)
    assert last is None


class TestListSymbols:
  def test_lists_symbols(self, data_dir):
    symbols = intraday.list_symbols('index', data_dir=data_dir)
    assert symbols == ['SPX']

  def test_empty_schema(self, data_dir):
    (data_dir / 'forex').mkdir()
    symbols = intraday.list_symbols('forex', data_dir=data_dir)
    assert symbols == []

  def test_missing_schema_returns_empty(self, data_dir):
    symbols = intraday.list_symbols('nonexistent', data_dir=data_dir)
    assert symbols == []


class TestSymbolPath:
  def test_returns_correct_path(self, data_dir):
    path = intraday.symbol_path('SPX', 'index', data_dir=data_dir)
    assert path.name == 'SPX.parquet'
    assert path.parent.name == 'index'


class TestWriteBars:
  def test_write_new_symbol(self, data_dir):
    times = pd.date_range('2024-06-01 09:30', periods=10, freq='min', tz='UTC')
    df = pd.DataFrame({
      'o': range(10), 'h': range(10), 'l': range(10), 'c': range(10),
      'v': range(10), 'a': range(10), 'bc': range(10),
    }, index=times, dtype=float)
    df.index.name = 'time'

    rows = intraday.write_bars(df, 'NEWSTOCK', 'stk', data_dir=data_dir)
    assert rows == 10

    read_back = intraday.get_bars(
      'NEWSTOCK', 'stk',
      datetime(2024, 6, 1, 9, 30, tzinfo=timezone.utc),
      datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
      data_dir=data_dir,
    )
    assert len(read_back) == 10

  def test_append_deduplicates(self, data_dir):
    start = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, 19, 30, tzinfo=timezone.utc)
    existing = intraday.get_bars('SPX', 'index', start, end, data_dir=data_dir)
    assert len(existing) == 600

    # Append overlapping + new data
    times = pd.date_range('2024-01-02 19:00', periods=60, freq='min', tz='UTC')
    rng = np.random.default_rng(99)
    new_data = pd.DataFrame({
      'o': rng.normal(4800, 10, 60),
      'h': rng.normal(4810, 10, 60),
      'l': rng.normal(4790, 10, 60),
      'c': rng.normal(4800, 10, 60),
      'v': rng.integers(100, 5000, 60).astype(float),
      'a': rng.normal(4800, 10, 60),
      'bc': rng.integers(10, 200, 60).astype(float),
    }, index=times)
    new_data.index.name = 'time'

    rows = intraday.write_bars(new_data, 'SPX', 'index', data_dir=data_dir)
    assert rows > 0

    total = intraday.get_bars(
      'SPX', 'index',
      datetime(2024, 1, 2, tzinfo=timezone.utc),
      datetime(2024, 1, 3, tzinfo=timezone.utc),
      data_dir=data_dir,
    )
    # 600 original (9:30-19:29) + 31 new (19:30-19:59), 29 overlap replaced
    assert len(total) == 630
