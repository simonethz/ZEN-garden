from zen_garden import Results, run
import os

os.chdir("C:/Users/simon/PycharmProjects/test")
run(dataset="C:/Users/simon/PycharmProjects/test/5_multiple_time_steps_per_year")



print(f"Baseline scenario ________________________________________")
base = Results(path='C:/Users/simon/PycharmProjects/test/outputs/5_multiple_time_steps_per_year_base')
capacity_CH_gas = base.get_total('capacity', index=("natural_gas_boiler", None, "CH"), year = 0).iloc[0,0]
capacity_DE_gas = base.get_total('capacity', index=("natural_gas_boiler", None, "DE"), year = 0).iloc[0,0]
print(f"German Capacity Gas Boiler: {capacity_DE_gas}")
print(f"Swiss Capacity Gas Boiler: {capacity_CH_gas}")
print(f"Total Capacity Gas Boiler: {capacity_DE_gas + capacity_CH_gas}")

capacity_CH_HP = base.get_total('capacity', index=("heat_pump", None, "CH"), year = 0).iloc[0,0]
capacity_DE_HP = base.get_total('capacity', index=("heat_pump", None, "DE"), year = 0).iloc[0,0]
print(f"German Capacity HP: {capacity_DE_HP}")
print(f"Swiss Capacity HP: {capacity_CH_HP}")
print(f"Total Capacity HP: {capacity_DE_HP + capacity_CH_HP}")

print(f"Tutorial 2 scenario 1) ________________________________________")
t21 = Results(path='C:/Users/simon/PycharmProjects/test/outputs/5_multiple_time_steps_per_year_Tutorial_2_1')
capacity_CH_HP = t21.get_total('capacity', index=("heat_pump", None, "CH"), year = 0).iloc[0,0]
capacity_DE_HP = t21.get_total('capacity', index=("heat_pump", None, "DE"), year = 0).iloc[0,0]
print(f"German Capacity HP: {capacity_DE_HP}")
print(f"Swiss Capacity HP: {capacity_CH_HP}")
print(f"Total Capacity HP: {capacity_DE_HP + capacity_CH_HP}")

print(f"Tutorial 2 scenario 2) ________________________________________")
t22 = Results(path='C:/Users/simon/PycharmProjects/test/outputs/5_multiple_time_steps_per_year_Tutorial_2_2')
capacity_CH_HP = t22.get_total('capacity', index=("heat_pump", None, "CH"), year = 0).iloc[0,0]
capacity_DE_HP = t22.get_total('capacity', index=("heat_pump", None, "DE"), year = 0).iloc[0,0]
print(f"German Capacity HP: {capacity_DE_HP}")
print(f"Swiss Capacity HP: {capacity_CH_HP}")
print(f"Total Capacity HP: {capacity_DE_HP + capacity_CH_HP}")

print(f"NEW ________________________________________")
new = Results(path='C:/Users/simon/PycharmProjects/test/outputs/5_multiple_time_steps_per_year')
capacity_CH_HP = new.get_total('capacity', index=("heat_pump", None, "CH"), year = 0).iloc[0,0]
capacity_DE_HP = new.get_total('capacity', index=("heat_pump", None, "DE"), year = 0).iloc[0,0]
print(f"German Capacity HP: {capacity_DE_HP}")
print(f"Swiss Capacity HP: {capacity_CH_HP}")
print(f"Total Capacity HP: {capacity_DE_HP + capacity_CH_HP}")

