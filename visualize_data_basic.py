from __future__ import annotations

import argparse
import sys
from collections import OrderedDict, deque, namedtuple
from pathlib import Path
from queue import Queue
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from genki_wave.callbacks import WaveCallback
from genki_wave.data import DataPackage, Euler3d, Point3d
from genki_wave.threading_runner import WaveListener
from matplotlib import animation
from more_itertools import flatten
from scipy.interpolate import interp1d


names_to_description = {
    "acc": "Total acceleration",
    "linacc": "Linear acceleration",
    "grav": "Gravity",
    "gyro": "Gyro",
    "euler": "Euler angles",
    "inference": "Model inference",
}


def normalize_ts(ts: np.ndarray, div_factor: int = 10**6) -> np.ndarray:
    """Assuming a monotonically increasing ts, shift it such that the last element is 0 and divide by a
    factor

    Examples:
        >>> normalize_ts(np.array([0.0, 10.0, 20.0, 20.0, 30.0, 50.0, 60.0]), div_factor=10)
        array([-6., -5., -4., -4., -3., -1.,  0.])
    """
    assert np.all(np.diff(ts) >= 0.0), "Expected ts to be monotonically increasing"
    ts = ts - ts[-1]
    ts = ts / div_factor
    return ts

def interp1d_np(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, axis=0, bounds_error=False, kind="linear") -> np.ndarray:
    """Interpolate each column (first dimension) of an array using linear interpolation

    NOTE: The default `axis=0` differs from the implementation of `interp_1d_torch` that interpolates over the last
          dimension.
    """
    f = interp1d(xp, fp, axis=axis, kind=kind, bounds_error=bounds_error)
    return f(x)


def generate_ts_support(signal_len: int, sampling_rate: int) -> np.ndarray:
    """Generate a signal based on given length and sampling rate, with the last sample = 0.0

    Args:
        signal_len: Length of the signal to generate
        sampling_rate: Sampling rate of the signal to generate

    Returns:
        A signal based on the given len and sampling rate

    Examples:
        >>> generate_ts_support(10, sampling_rate=10)
        array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0. ])
    """
    return np.linspace(1 / sampling_rate - signal_len / sampling_rate, 0, int(signal_len), endpoint=True)

class CircularBuffer:
    """Handle stream of incoming data from Wave using a circular buffer for each signal

    Args:
        signals: The names of the signals to store in the buffer
        maxlen: Maximum length of the buffer for each signal, needs to be longer than `fetch_x_sec * sampling_rate`
        fetch_x_sec: How many seconds of data to fetch each time `get_elements` is called
        sampling_rate: The sampling rate of the incoming signal
    """

    TIMESTAMP_COL = "timestamp_us"
    NUM_US_IN_S = 10**6

    def __init__(self, signals: List[str], maxlen: int, fetch_x_sec: float, sampling_rate: int):
        if not (maxlen >= fetch_x_sec * sampling_rate):
            raise ValueError(
                f"The buffer needs to be larger than the amount of data fetched, "
                f"got {maxlen=} and {fetch_x_sec * sampling_rate=}"
            )
        signals_w_timestamp = signals + [self.TIMESTAMP_COL]

        self._fetch_x_sec = fetch_x_sec
        self._num_samples_to_show = int(self._fetch_x_sec * sampling_rate)

        self._signals = OrderedDict((s, deque(maxlen=maxlen)) for s in signals_w_timestamp)
        self._ts_support = generate_ts_support(signal_len=self._num_samples_to_show, sampling_rate=sampling_rate)
        self._last_time = None

    @property
    def signals(self):
        return OrderedDict([(key, value) for key, value in self._signals.items() if key != self.TIMESTAMP_COL])

    def assert_len(self):
        """Check if all the buffers have the same number of elements"""
        sig_first = next(iter(self._signals.values()))
        for sig in self._signals.values():
            assert len(sig_first) == len(sig)

    def __len__(self):
        self.assert_len()
        sig_first = next(iter(self._signals.values()))
        return len(sig_first)

    def extend(self, attr_name: str, vals: List[DataPackage]):
        data_packages = [v for v in vals if isinstance(v, DataPackage)]
        data = [to_tuple(getattr(v, attr_name)) for v in data_packages]
        self._signals[attr_name].extend(data)

    def extend_all(self, vals: List[DataPackage]):
        for sig_name in self._signals:
            self.extend(sig_name, vals)
        self.assert_len()

    def get_elements(self):
        ts_org = self._signals[self.TIMESTAMP_COL]
        ts = normalize_ts(np.array(ts_org), self.NUM_US_IN_S)
        idxs = ts > -self._fetch_x_sec

        signals_output = OrderedDict([(key, np.array(value)[idxs]) for key, value in self.signals.items()])

        ts = ts[idxs]
        signals_interp_output = []
        for key, value in signals_output.items():
            value_interp = interp1d_np(self._ts_support, ts, value)
            signals_interp_output.append((key, value_interp))

        signals_interp_output = OrderedDict(signals_interp_output)

        return self._ts_support, signals_interp_output


def to_tuple(p: Point3d | int) -> tuple | int:
    """Unpack dataclasses into a tuple"""
    if isinstance(p, Point3d):
        return p.x, p.y, p.z
    elif isinstance(p, Euler3d):
        return p.roll, p.pitch, p.yaw
    return p


def find_max_abs_val(x: np.ndarray) -> float:
    """Find max(abs(x)) ignoring nan and inf values"""
    x_flat = x.ravel()
    x_flat = x_flat[~(np.isinf(x_flat) | np.isnan(x_flat))]
    return np.max(np.abs(x_flat))


class AddToQueue(WaveCallback):
    def __init__(self, queue: Queue):
        self.q = queue

    def _data_handler(self, data: DataPackage) -> None:
        self.q.put(data)

    def _button_handler(self, data) -> None:
        pass


def main(
    ble_address: str,
    signal_names: List[str],
    fetch_x_sec: int,
    sampling_rate: int,
    animation_interval: int,
):
    buffer = CircularBuffer(
        signal_names,
        maxlen=2 * sampling_rate * fetch_x_sec,
        fetch_x_sec=fetch_x_sec,
        sampling_rate=sampling_rate,
    )
    AxLines = namedtuple("AxLines", "ax lines")

    queue = Queue(maxsize=2 * sampling_rate * fetch_x_sec)
    cb = AddToQueue(queue)
    with WaveListener(ble_address, [cb]):
        fig, axes = plt.subplots(nrows=len(buffer.signals), ncols=1, figsize=(10, 10))

        if isinstance(axes, plt.Axes):
            axes = np.array([axes])

        # Plot all zeros in the beginning for initialization and grab the line objects
        ax_lines_list = []
        for k in range(len(axes)):
            lines_for_ax = axes[k].plot(buffer._ts_support, np.zeros((len(buffer._ts_support), 3)))
            ax_lines_list.append(AxLines(ax=axes[k], lines=lines_for_ax))

        # Initialize the plot
        for ax_lines, key in zip(ax_lines_list, buffer.signals):
            ax_lines.ax.set_ylim([-1, 1])
            ax_lines.ax.set_xlim([-fetch_x_sec, 0])
            ax_lines.ax.set_title(names_to_description[key])

        # Remove x-ticks from all plots except the bottom plot
        for ax_lines in ax_lines_list[:-1]:
            ax_lines.ax.set_xticks([])

        # Set labels for legend
        idx_to_dim = ["x", "y", "z"]
        for i in range(len(ax_lines_list)):
            for j in range(len(ax_lines_list[i].lines)):
                ax_lines_list[i].lines[j].set_label(idx_to_dim[j])

        def animate(_):
            new_data = []
            while not queue.empty():
                new_data.append(queue.get())
            buffer.extend_all(new_data)

            if len(buffer) > 1:
                ts, signals_all = buffer.get_elements()
                for i in range(len(ax_lines_list)):
                    signals = list(signals_all.values())[i]
                    for j in range(len(ax_lines_list[i].lines)):
                        ax_lines_list[i].lines[j].set_ydata(signals[:, j])

                    max_val = find_max_abs_val(signals)
                    max_val = 1.0 if max_val == 0.0 else max_val
                    max_val = max_val * 1.1
                    ax_lines_list[i].ax.set_ylim([-max_val, max_val])

            return list(flatten([al.lines for al in ax_lines_list])) + [
                al.ax.legend(loc="lower left") for al in ax_lines_list
            ]

        ani = animation.FuncAnimation(fig, animate, interval=animation_interval)  # noqa: F841
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ble_address", type=str)
    parser.add_argument("--signal-names", nargs="+", default=["acc", "gyro"])
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=95,
        help="Sampling rate used to determine the support of the visualizations",
    )
    parser.add_argument("--fetch-x-sec", type=int, default=4)
    parser.add_argument(
        "--animation-interval", type=int, default=25, help="The number of ms between re-drawing the plot"
    )
    args = parser.parse_args()

    main(
        args.ble_address,
        args.signal_names,
        args.fetch_x_sec,
        args.sampling_rate,
        args.animation_interval,
    )
