"""
@Author: Lin
@Version: 1.4
@Date: 30/07/2024 - 12/08/2024
@Description:
    1.0 Entrance -
        User order info
        Coffee machine behaviour simulation, includes all components
    1.1 Add modify_pump_off_method
        pump won't stop forever
    1.2 Extend CoffeeMachine.py
        generate csv, using cof_level, milk_level, con_level, timestamp, pump_is_turn_on as columns
    1.3 Integrate all components into CoffeeMachine.py.
        Modify variable pump_is_turn_on to be a global variable
        Delete modify_pump_off_method
        update the attack method, pump will be turned on/off irregularly
        train LSTM model and evaluate the performance
    1.4 integrate LSTM model to detect
"""
import random
import csv
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model('Detection/Model/lstm_model.h5')

# global variable
cof_level = 100
milk_level = 100
con_level = 100
original_temperature = 25.0  # original temp is 25 degrees
target_temperature = 80.0  # target temperature is 80 degrees
pump_is_turn_on = False  # state of the pump


# attack method
def attack_pump():
    global pump_is_turn_on
    pump_is_turn_on = True if random.random() < 0.3 else False  # 30% chance of attack


# Preprocess input data for prediction
def preprocess_input(cof, milk, con,timestamp):
    # Convert timestamp to datetime features
    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute

    # Create the input array including time features
    data = np.array([[cof, milk, con, year, month, day, hour, minute]])
    return data.reshape((data.shape[0], 1, data.shape[1]))  # Reshape to match model input


# Use the model to detect an attack
def detect_attack(cof, milk, con, timestamp):
    input_data = preprocess_input(cof, milk, con, timestamp)
    prediction = model.predict(input_data)

    # Post-process the prediction, e.g., apply a threshold to determine if an attack is detected
    predicted_cof, predicted_milk, predicted_con, predicted_year, predicted_month,\
        predicted_day, predicted_hour, predicted_minute = prediction[0]

    # thresholds
    cof_diff = abs(cof - predicted_cof)
    milk_diff = abs(milk - predicted_milk)
    con_diff = abs(con - predicted_con)
    if cof_diff < 10 or milk_diff < 10 or con_diff < 10:
        return True
    return False


# generate original and detection csv file
def generate_csv(data_type='original'):
    predicted_attack = False
    if data_type == 'original':
        csv_headers = ['Timestamp', 'cof_level', 'milk_level', 'con_level', 'pump_is_turn_on']
        dataset_dir = os.path.join('Detection', 'Dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        csv_file = os.path.join(dataset_dir, 'drink_level_log.csv')
    elif data_type == 'detection':
        csv_headers = ['Timestamp', 'cof_level', 'milk_level', 'con_level', 'predicted_attack', 'actual_attack']
        result_dir = os.path.join('Detection', 'Result')
        os.makedirs(result_dir, exist_ok=True)
        csv_file = os.path.join(result_dir, 'detection_results.csv')

    # Create CSV file if not exists
    if not os.path.isfile(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_headers)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if data_type == 'original':
        row = [timestamp, round(cof_level, 1), round(milk_level, 1), round(con_level, 1), pump_is_turn_on]
    elif data_type == 'detection':
        # Detect attack using the model
        predicted_attack = detect_attack(cof_level, milk_level, con_level, timestamp)
        row = [timestamp, round(cof_level, 1), round(milk_level, 1), round(con_level, 1), predicted_attack,
               pump_is_turn_on]

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    return predicted_attack


# user order
def order():
    # 1 kopi_ko song  2 kopi_c  3 kopi
    order_num = random.randint(1, 3)
    if order_num == 1:
        print("Order kopi_kosong(made by coffee)...")
    if order_num == 2:
        print("Order kopi_c(made by coffee & milk)...")
    if order_num == 3:
        print("Order kopi(made by coffee & condensed milk)...")
    return order_num


# Temperature sensor
def measure_current_temperature():
    # simulate to measure temp
    original_temperature = round(random.uniform(25.0, 90.0), 1)
    return original_temperature


# Liquid level sensor
def get_drink_level():
    print("Checking the current ingredient value now...!")
    print(f"The current value of coffee is {round(cof_level, 1)}")
    print(f"The current value of milk is {round(milk_level, 1)}")
    print(f"The current value of condensed milk is {round(con_level, 1)}")
    print("---------------------------------")
    return cof_level, milk_level, con_level


# Heater
def heat_up_from(current_temperature):
    while current_temperature < target_temperature:
        current_temperature += random.uniform(1.5, 3)
        print(f"Heating up to {round(current_temperature, 1)} degrees...")
    print(f"Heating Completed!!!")


# Pump
def pump_on():
    global pump_is_turn_on
    pump_is_turn_on = True
    print("Pump turns on, transferring ingredients to the cup...")


def pump_off():
    global pump_is_turn_on
    pump_is_turn_on = False
    print("Adding ingredients successfully! Pump turns off...")


# Coffee machine function
def CoffeeMachine():
    global cof_level
    global milk_level
    global con_level
    # Calculate total attacks and predicted attacks
    total_attacks_num = 0
    predicted_attacks_num = 0
    # Calculate FPR and FDR
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for _ in range(1000):
        # user order
        order_num = order()
        print("---------------start------------------")

        # Simulate pump attack
        attack_pump()

        # liquid level sensor check the current value of ingredients, refill if needed
        current_cof_level, current_milk_level, current_con_level = get_drink_level()

        # refill if needed
        if current_cof_level < 10 or current_milk_level < 10 or current_con_level < 10:
            print("Refill ingredients...")
            if current_cof_level < 10:
                cof_level = 100
            if current_milk_level < 10:
                milk_level = 100
            if current_con_level < 10:
                con_level = 100
            print("Refill ingredients completely!!")
        else:
            print("The ingredients are sufficient, no need to refill!!")
        print("---------------------------------")

        # temperature sensor measures current temperature
        print("Measuring coffee temperature...")
        current_temperature = measure_current_temperature()
        print(f"The current temperature is {current_temperature} degrees...")
        print("---------------------------------")
        if current_temperature < target_temperature:
            print("Start to heat...")
            # heater is heating
            heat_up_from(current_temperature)
        else:
            print("No need to heat...")
        print("---------------------------------")

        # pump transports ingredients
        # default is in close
        if not pump_is_turn_on:
            pump_on()
            # decrease the value of ingredients
            if order_num == 1:
                cof_level -= 10
            if order_num == 2:
                cof_level -= 10
                milk_level -= 10
            if order_num == 3:
                cof_level -= 10
                con_level -= 10
            pump_off()
            # Log data to original CSV
            generate_csv('original')
        else:
            print("Pump is already on, reducing ingredient levels continuously...")
            total_attacks_num += 1
            while pump_is_turn_on:
                # reduce drink level non-regularly
                cof_level -= random.uniform(1, 9)
                milk_level -= random.uniform(1, 9)
                con_level -= random.uniform(1, 9)
                # Check if any level has reached or dropped below zero
                if cof_level < 10 or milk_level < 10 or con_level < 10:
                    break
            # log to original CSV
            generate_csv('original')

        # collect detection data
        predicted_attacks = generate_csv('detection')

        if predicted_attacks:
            predicted_attacks_num += 1

        if pump_is_turn_on and predicted_attacks:
            true_positive += 1
        elif pump_is_turn_on and not predicted_attacks:
            false_negative += 1
        elif not pump_is_turn_on and predicted_attacks:
            false_positive += 1
        elif not pump_is_turn_on and not predicted_attacks:
            true_negative += 1
        # Calculate metrics
        if (false_positive + true_negative) > 0:
            fpr = false_positive / (false_positive + true_negative)
        else:
            fpr = 0

        if (false_positive + true_positive) > 0:
            fdr = false_positive / (false_positive + true_positive)
        else:
            fdr = 0
        # end
        print("----------------end-----------------")

    print(f"Total Actual Attacks: {total_attacks_num}")
    print(f"Total Predicted Attacks: {predicted_attacks_num}")
    print(f"False Positive Rate (FPR): {fpr:.2f}")
    print(f"False Discovery Rate (FDR): {fdr:.2f}")


if __name__ == "__main__":
    CoffeeMachine()
