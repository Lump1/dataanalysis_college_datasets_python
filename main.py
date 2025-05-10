import graphics as gp


def main():
    # gp.task_1("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
    # gp.task_2("https://www.meteoblue.com/en/weather/archive/export?daterange=2024-01-01+-+2025-03-30&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&submit_csv=CSV&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=%2B00%3A00&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30")
    # gp.task_3("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
    # gp.task_3_2("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/wisc_bc_data.csv")
    # gp.task_3_3("https://www.meteoblue.com/en/weather/archive/export?daterange=2024-01-01+-+2025-03-31&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&submit_csv=CSV&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=%2B00%3A00&timeResolution=daily&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=30%3B50&gddBase=10&gddLimit=30")
    # gp.task_4("https://raw.githubusercontent.com/x5or6/A-Visual-History-of-Nobel-Prize-Winners/master/archive.csv")
    # gp.task_5()

    while True:
        print('Select option below \n 1: Task 1 \n 2: Task 2 \n 3.1: Task 3, 1 subtask \n 3.2: Task 3, 2 subtask \n 3.3: Task 3, 3 subtask\n 4: Task 4\n 5: Task 5\n indz1: Task indz 1\n indz2: Task indz 2\n 4pract: 4th pract 2nd task\n 6: Exit\n')
        selectedOption = input("Ur choise: ")
        match selectedOption:
            case '1':
                gp.task_1(
                    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
            case '2':
                gp.task_2("https://www.meteoblue.com/en/weather/archive/export?daterange=2024-01-01+-+2025-03-30&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&submit_csv=CSV&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=%2B00%3A00&timeResolution=hourly&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=10%3B30&gddBase=10&gddLimit=30")
            case "3.1":
                gp.task_3(
                    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")
            case "3.2":
                gp.task_3_2(
                    "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/wisc_bc_data.csv")
            case "3.3":
                gp.task_3_3("https://www.meteoblue.com/en/weather/archive/export?daterange=2024-01-01+-+2025-03-31&locations%5B%5D=basel_switzerland_2661604&domain=ERA5T&submit_csv=CSV&params%5B%5D=&params%5B%5D=temp2m&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&params%5B%5D=&utc_offset=%2B00%3A00&timeResolution=daily&temperatureunit=CELSIUS&velocityunit=KILOMETER_PER_HOUR&energyunit=watts&lengthunit=metric&degree_day_type=30%3B50&gddBase=10&gddLimit=30")
            case '4':
                gp.task_4(
                    "https://raw.githubusercontent.com/x5or6/A-Visual-History-of-Nobel-Prize-Winners/master/archive.csv")
            case '5':
                gp.task_5()
            case 'indz1':
                gp.task_indz_1("titanic.csv")
            case 'indz2':
                gp.task_indz_2("Seed_Data.csv")
            case 'indz2':
                gp.task_indz_2("Seed_Data.csv")
            case '4pract':
                gp.task_2_pract_4("winequality-red.csv")
            case "6":
                break
            case _:
                print("Wrong option! Retry again!")


if __name__ == "__main__":
    main()
