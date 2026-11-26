import json
from pathlib import Path


APPLIANCE_LOAD_LABELS = {
    "Air Conditioner": "RLC",
    "Compact Fluorescent Lamp": "RC",
    "Fan": "RL",
    "Fridge": "RL",
    "Hairdryer": "RL",
    "Heater": "R",
    "Incandescent Light Bulb": "R",
    "Laptop": "RC",
    "Microwave": "RLC",
    "Vacuum": "RL",
    "Washing Machine": "RLC",
}


def main():
    directory = Path(__file__).parent
    with (directory / "meta.json").open(encoding="utf-8") as file:
        records = json.load(file)

    for record in records:
        appliance = record["meta"]["appliance"]
        if appliance["load"] == "NL":
            appliance_type = appliance["type"]
            if appliance_type not in APPLIANCE_LOAD_LABELS:
                raise ValueError(
                    f"No label mapping for appliance type "
                    f"{appliance_type!r} in record {record.get('id')!r}"
                )
            appliance["load"] = APPLIANCE_LOAD_LABELS[appliance_type]

    output_path = directory / "meta_labeled.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=4, ensure_ascii=False)
        file.write("\n")


if __name__ == "__main__":
    main()