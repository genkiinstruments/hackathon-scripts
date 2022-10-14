import argparse
import time
from pathlib import Path

from genki_wave.callbacks import CsvOutput
from genki_wave.threading_runner import WaveListener


def main(ble_address: str, output_filepath: str) -> None:
    cb = CsvOutput(Path(output_filepath))
    with WaveListener(ble_address, [cb]):
        while True:
            time.sleep(1.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ble_address", type=str)
    parser.add_argument("output_filepath", type=str)
    args = parser.parse_args()

    main(args.ble_address, args.output_filepath)
