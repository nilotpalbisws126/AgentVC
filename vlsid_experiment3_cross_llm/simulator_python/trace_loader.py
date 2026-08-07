"""
Trace Loader Module for Vehicular Cloud Task Scheduling.
Parses both synthetic traces (car_data.txt, task_data.txt, trace_data_1..9)
and real car park traces (Queens Anne, Grand Arcade).
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

@dataclass
class CarTrace:
    car_id: int
    arrival_time: int
    departure_time: int
    car_type: str
    expected_stay: int = field(init=False)

    def __post_init__(self):
        self.expected_stay = self.departure_time - self.arrival_time

@dataclass
class TaskTrace:
    task_id: int
    arrival_time: int
    execution_time: int
    deadline: int
    price: int
    packet_id: int = 0
    packet_count: int = 1

class TraceLoader:
    @staticmethod
    def load_cars(file_path: str) -> List[CarTrace]:
        cars = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Car trace file not found: {file_path}")
        
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    c_id = int(parts[0])
                    arr = int(parts[1])
                    dep = int(parts[2])
                    raw_type = parts[3]
                    if raw_type == "0" or raw_type == "S":
                        c_type = "S"
                    elif raw_type == "1" or raw_type == "M":
                        c_type = "M"
                    else:
                        c_type = "L"
                    cars.append(CarTrace(c_id, arr, dep, c_type))
        return cars

    @staticmethod
    def load_tasks(file_path: str) -> List[TaskTrace]:
        tasks = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Task trace file not found: {file_path}")

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    t_id = int(parts[0])
                    arr = int(parts[1])
                    exec_t = int(parts[2])
                    dl = int(parts[3])
                    price = int(parts[4])
                    pkt_id = int(parts[5]) if len(parts) > 5 else 0
                    pkt_count = int(parts[6]) if len(parts) > 6 else 1
                    tasks.append(TaskTrace(t_id, arr, exec_t, dl, price, pkt_id, pkt_count))
        return tasks
