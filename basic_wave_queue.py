import argparse
import time
from collections import deque

from typing import Optional

from genki_wave.callbacks import WaveCallback
from genki_wave.data import ButtonEvent, DataPackage
from genki_wave.threading_runner import WaveListener


class CollectIntoQueueCallback(WaveCallback):
    def __init__(self, maxlen: Optional[int] = None):
        self.q = deque(maxlen=maxlen)

    def _data_handler(self, data: DataPackage) -> None:
        self.q.append(data.acc.x)

    def _button_handler(self, data: ButtonEvent) -> None:
        pass

    def pop(self, n):
        num_to_pop = min(n, len(self.q))
        return [self.q.popleft() for _ in range(num_to_pop)]

    def pop_all(self):
        return self.pop(len(self.q))

    def view_all(self):
        return list(self.q)


def main(ble_address: str) -> None:
    cb = CollectIntoQueueCallback()
    with WaveListener(ble_address, [cb]):
        while True:
            time.sleep(2.0)
            print(cb.pop_all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ble_address", type=str)
    args = parser.parse_args()

    main(args.ble_address)
