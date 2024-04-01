import json

network_settings = {
    "AboveNet": {
        "Propagation_delay_ms": 5e-4,
        "Capacity_Gbps": 1,
        "Number_of_overlay_nodes": 15,
        "Background_utilization": [30, 50]
    },
    "Hexagonal": {
        "Propagation_delay_ms": 5e-4,
        "Capacity_Gbps": 0.5,  # assuming 'tunable' is represented by the initial value
        "Number_of_overlay_nodes": 10,
        "Background_utilization": [30, 50]
    },
    "Roofnet": {
        "Propagation_delay_ms": 5e-4,
        "Capacity_Gbps": 0.001,
        "Number_of_overlay_nodes": 10,  # assuming the current value as mentioned in the comment
        "Background_utilization": [30, 50]
    }
}

# Writing JSON data
with open('./network_settings.json', 'w') as json_file:
    json.dump(network_settings, json_file, indent=4)
