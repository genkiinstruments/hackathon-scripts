# Hackathon files
Some scripts to get you started using and interfacing with Wave

## Installation
```
pip install -r requirements.txt
```

## How to use
Start by turning on your Wave. It should have 2 dots that fade in and out of view. Then run:
```
python discover_ble_address.py
```

this will print a list of bluetooth addresses for available Waves.

Let's say your address is `abcd-1234`. Then you can run the script as follows

```
python basic_wave_print.py abcd-1234 --print-data --print-button
```

```
python basic_wave_queue.py abcd-1234
```

```
python basic_wave_csv.py abcd-1234 output_file.csv
```
