"""
Prepare chronological PyTorch inputs and three-class targets.
X: CPU float32 tensors [sample, lookback, ticker, feature].
y: CPU int64 tensors [sample, ticker]; 0 negative, 1 neutral, 2 positive.
Target: open[t+horizon] / open[t] - 1. Equality at a deadzone bound is neutral.
No log transforms. StandardScaler is fitted only on unique training input days,
separately per ticker/feature. Targets are never scaled.
Split eligible prediction dates chronologically, then purge training samples
whose target dates reach the first test prediction date.

Congress: seven daily aggregates (counts, lower/upper dollar totals, mean delay).
BioID/action embeddings are not provided by this aggregate-only interface.
Noise: daily congressional vectors resampled from training input days only;
preserves joint daily values, not their original timing/serial dependence.
For repeated tuning, the second partition is validation, not an untouched test.
Dependencies: numpy, pandas, scikit-learn, torch, openpyxl for XLSX.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MARKET_FIELDS = ("Current Open", "Previous Open", "Previous High",
				 "Previous Low", "Previous Close", "Previous Volume")

CONGRESS_FIELDS = ("purchase_count", "sale_count", "low_purchase_value",
				   "high_purchase_value", "low_sale_value", "high_sale_value",
				   "mean_delay_days")

CONDITIONS = ("market", "congress", "noise")

PURCHASE_ACTIONS = ("purchase",)

SALE_ACTIONS = ("sale", "sale (full)", "sale (partial)")


def _table(path):
	# Please don't load non csv/excel files, your's truly, Hunter from 10/12/2026
	# The time it took to write these comments probably could have implemented logic to handle errors like this, Hunter from 10/13/2026
	path = Path(path)
	suffix = path.suffix.lower()
	# Updated to throw proper errors when I ignore my instructions. Hunter from 10/13/2026 - 5 minutes later
	if suffix == ".csv":
		return pd.read_csv(path)
		
	if suffix in (".xlsx", ".xls"):
		return pd.read_excel(path)  # XLS requires xlrd; XLSX uses openpyxl.
		
	raise ValueError(f"Unsupported file type: {suffix}")


def _lists(value):
	if isinstance(value, list):
		return value
		
	if pd.isna(value):
		return []
		
	result = json.loads(value)
	if not isinstance(result, list):
		raise ValueError("Disclosure cells must contain JSON lists")
		
	return result


def _amount(value):
	text = str(value).replace(",", "").strip()
	match = re.fullmatch(r"\$?\s*(\d+(?:\.\d+)?)\s*[-–—]\s*\$?\s*(\d+(?:\.\d+)?)", text)
	exact = re.fullmatch(r"\$?\s*(\d+(?:\.\d+)?)", text)
	if exact is not None:
		amount = float(exact.group(1))
		if not np.isfinite(amount):
			raise ValueError(f"Invalid dollar amount: {value!r}")
		return amount, amount
		
	low, high = map(float, match.groups())
	if not np.isfinite([low, high]).all() or high < low:
		raise ValueError(f"Invalid dollar range: {value!r}")
		
	return low, high


def _dates(values, name):
	dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
	if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
		raise ValueError(f"{name} must be valid, unique and sorted")
		
	if dates.tz is not None or not dates.equals(dates.normalize()):
		raise ValueError(f"{name} must be timezone-free daily dates")
		
	return dates


def _congress(frame, dates, tickers):
	#Assign disclosures to the first market day strictly after filing.
	filing = _dates(frame["Filed Date"], "Filed Date")
	result = np.zeros((len(dates), len(tickers), len(CONGRESS_FIELDS)), dtype=float)
	delays = np.zeros(result.shape[:2])
	counts = np.zeros(result.shape[:2])
	
	for row, filed in zip(frame.to_dict("records"), filing):
		destination = dates.searchsorted(filed, side="right")
		
		for j, ticker in enumerate(tickers):
			ids, trades, actions, values = [
				_lists(row[f"{ticker} {field}"])
				for field in ("BioIDs", "Trade Dates", "Actions", "Values")
			]
			if len({len(ids), len(trades), len(actions), len(values)}) != 1:
				raise ValueError(f"Mismatched disclosure lists: {filed}, {ticker}")

			# Dropping IDs for now, because I was unable to conceptualize how to standardize the dimensions
			# Of the resulting tensor when there is variable number of Ids involved in every disclosure
			for trade, action, value in zip(trades, actions, values):
				trade_date = pd.Timestamp(trade)
				if pd.isna(trade_date):
					raise ValueError("Missing transaction date")
					
				delay = (filed - trade_date).days
				
				if delay < 0:
					# Raise ValueError("Transaction date follows filing date")
					# Found a total of 11 occurences in the source data, opted to disregard them for now.
					continue 
				action = action.strip().casefold()
				if action in PURCHASE_ACTIONS:
					count_column = CONGRESS_FIELDS.index("purchase_count")
					low_column = CONGRESS_FIELDS.index("low_purchase_value")
					high_column = CONGRESS_FIELDS.index("high_purchase_value")
				elif action in SALE_ACTIONS:
					count_column = CONGRESS_FIELDS.index("sale_count")
					low_column = CONGRESS_FIELDS.index("low_sale_value")
					high_column = CONGRESS_FIELDS.index("high_sale_value")
				elif action == "exchange":
					# Meaning unresolved; exclude from purchase/sale aggregates.
					continue
				else:
					raise ValueError(f"Unknown action {action!r}; specify its meaning before processing")
					
				low, high = _amount(value)
				if destination == len(dates):
					continue
				if filed < dates[0]:
					continue
					
				result[destination, j, count_column] += 1
				result[destination, j, low_column] += low
				result[destination, j, high_column] += high
				delays[destination, j] += delay
				counts[destination, j] += 1
				
	result[:, :, CONGRESS_FIELDS.index("mean_delay_days")] = np.divide(
																delays,
																counts,
																out=np.zeros_like(delays),
																where=counts > 0)
	return result


def _standardize(data, training_days):
	# Flatten ticker/feature axes for sklearn, then restore their shape.
	shape = data.shape
	flat = data.reshape(shape[0], -1)
	scaler = StandardScaler()
	scaler.fit(flat[training_days])
	return scaler.transform(flat).reshape(shape), scaler


def _prepare(condition, test_size, lookback, horizon, deadzone, scale, data_dir, start, noise_seed):
	if condition not in CONDITIONS:
		raise ValueError(f"condition must be one of {CONDITIONS}")
		
	for name, value in (("lookback", lookback), ("horizon", horizon)):
		if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
			raise ValueError(f"{name} must be a positive integer")
			
	if not np.isfinite(deadzone) or deadzone < 0:
		raise ValueError("deadzone must be finite and nonnegative")
		
	if not isinstance(scale, bool):
		raise ValueError("scale must be True or False")
		
	data_dir = Path(data_dir)
	frame = _table(data_dir / "market_daily_decision_data.xlsx")
	all_dates = _dates(frame.pop("Decision Date"), "Decision Date")
	# Pulls all the prefix ticker names from the market data headers
	tickers = tuple(sorted(c.removesuffix(" Current Open") for c in frame
						   if c.endswith(" Current Open")))
	if not tickers:
		raise ValueError("No ticker Current Open columns")

	# Ensures the [dats, 20, 6] shape remains correct
	columns = [f"{ticker} {field}" for ticker in tickers for field in MARKET_FIELDS]
	raw = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(float)

	# finds the end of leading market days which lack a complete (all tickers present) coverage
	complete = np.isfinite(raw).all(axis=1)
	if start is None:
		if not complete.any():
			raise ValueError("No complete market day")
			
		first = np.flatnonzero(complete)[0]
	else:
		first = all_dates.searchsorted(pd.Timestamp(start), side="left")
	if first == len(all_dates) or not complete[first:].all():
		raise ValueError("Missing/nonfinite market values after start; choose a complete start date")

	# trims the dates to the range starting at the found start of complete market days.
	dates = all_dates[first:]
	market = raw[first:].reshape(len(dates), len(tickers), len(MARKET_FIELDS))
	if (market[:, :, :5] <= 0).any() or (market[:, :, 5] < 0).any():
		raise ValueError("Prices must be positive and volumes nonnegative")
		
	eligible = np.arange(lookback - 1, len(dates) - horizon)
	# Split prediction-date indices, preserving order across every ticker.
	train_idx, test_idx = train_test_split(eligible, test_size=test_size, shuffle=False)
	train_idx = train_idx[train_idx + horizon < test_idx[0]]
	if not len(train_idx):
		raise ValueError("No training samples remain after horizon purging")
		
	# Fit once per unique training input day, not once per repeated lag.
	training_days = np.zeros(len(dates), dtype=bool)
	for i in train_idx:
		training_days[i - lookback + 1:i + 1] = True
		
	congress_scaler = market_scaler = None
	extra = None
	
	if condition != "market":
		# Align on the full calendar before removing incomplete market history.
		congress = _congress(_table(data_dir / "congressional_disclosures_wide.xlsx"),
							 all_dates, tickers)[first:]
		if condition == "noise":
			rng = np.random.default_rng(noise_seed)
			pool = congress[training_days]
			extra = pool[rng.integers(len(pool), size=len(dates))]
		else:
			extra = congress
			
		if scale:
			_, congress_scaler = _standardize(congress, training_days)
			extra = congress_scaler.transform(extra.reshape(len(dates), -1)).reshape(extra.shape)
			
	if scale:
		market, market_scaler = _standardize(market, training_days)
		
	features = market if extra is None else np.concatenate([market, extra], axis=-1)
	# Use original prices for labels, regardless of input scaling.
	opens = raw[first:].reshape(len(dates), len(tickers), len(MARKET_FIELDS))[:, :, 0]
	
	def batch(indices):
		X = np.stack([
			features[i - lookback + 1:i + 1] 
			for i in indices
		]).astype(np.float32)

		# fuuuuuutuuuuure, careful not to leak it
		returns = opens[indices + horizon] / opens[indices] - 1
		
		y = np.where(
			returns < -deadzone, 0, # lose value
			np.where(
				returns > deadzone, 2, # gain value
				1 #neutral value
			)
		).astype(np.int64)
		if not np.isfinite(X).all():
			raise ValueError("Features exceed float32 range")
			
		return X, y
		
	X_train, y_train = batch(train_idx)
	X_test, y_test = batch(test_idx)
	
	info = {
		"tickers": tickers,
		"feature_names": MARKET_FIELDS + (() if extra is None else CONGRESS_FIELDS),
		"train_dates": dates[train_idx].to_numpy(),
		"train_target_dates": dates[train_idx + horizon].to_numpy(),
		"test_dates": dates[test_idx].to_numpy(),
		"test_target_dates": dates[test_idx + horizon].to_numpy(),
		"scaler_fit_dates": dates[training_days].to_numpy(),
		"market_scaler": market_scaler, "congress_scaler": congress_scaler,
		"settings": {"condition": condition, "test_size": test_size, "lookback": lookback,
					 "horizon": horizon, "deadzone": deadzone, "scale": scale,
					 "start": str(dates[0].date()), "noise_seed": noise_seed},
	}
	return (X_train, y_train, X_test, y_test), info


def get_xyts(condition="market", test_size=0.2, lookback=63, horizon=5,
			 deadzone=0.005, scale=True, *, data_dir=None, start=None,
			 noise_seed=42, return_info=False):
	"""Return X_train, y_train, X_test, y_test as CPU tensors.

	data_dir defaults to this module's adjacent data/ directory.
	start=None trims leading incomplete market history; later missing data fails.
	test_size is a fraction or integer count supported by train_test_split.
	scale=False preserves raw feature values (apart from conversion to float32).
	return_info=True appends a fifth result with dates, tickers and fitted scalers.
	Keep all settings, including noise_seed, identical for matched comparisons.
	"""
	import torch
	if data_dir is None:
		data_dir = Path(__file__).resolve().parent / "data"
	arrays, info = _prepare(condition, test_size, lookback, horizon, deadzone,
							scale, data_dir, start, noise_seed)
	result = tuple(torch.tensor(a, dtype=torch.float32 if i % 2 == 0 else torch.long,
								device="cpu") for i, a in enumerate(arrays))
	return (*result, info) if return_info else result
