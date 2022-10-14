import argparse
import time

from pprint import pprint

from genki_wave.callbacks import WaveCallback
from genki_wave.data import ButtonEvent, DataPackage
from genki_wave.threading_runner import WaveListener


class PrintingCallback(WaveCallback):
    def __init__(self, print_data: bool, print_button: bool):
        self.print_data = print_data
        self.print_button = print_button

    def _data_handler(self, data: DataPackage) -> None:
        if self.print_data:
            pprint("A data package:")
            pprint(data)
            pprint("The same package as a dict:")
            pprint(data.as_dict())

    def _button_handler(self, data: ButtonEvent) -> None:
        if self.print_button:
            pprint("A button package:")
            pprint(data)


def main(ble_address: str, print_data: bool, print_button: bool) -> None:
    cb = PrintingCallback(print_data=print_data, print_button=print_button)
    with WaveListener(ble_address, [cb]):
        while True:
            time.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ble_address", type=str)
    parser.add_argument("--print-data", action="store_true")
    parser.add_argument("--print-button", action="store_true")
    args = parser.parse_args()

    main(args.ble_address, args.print_data, args.print_button)
