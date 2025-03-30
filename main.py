import matplotlib.pyplot as plt
import seaborn as sns
import loaddata as ld

def main():
    sns.set_style("whitegrid")
    dataset = ld.loadDataset("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")

    print(dataset)

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=dataset, x="bmi", y="charges", hue="smoker", 
        palette={"yes": "red", "no": "blue"}, alpha=0.6
    )

    plt.title("Залежність між ІМТ та медичними витратами з урахуванням статусу курця")
    plt.xlabel("Індекс маси тіла (BMI)")
    plt.ylabel("Медичні витрати (USD)")
    plt.legend(title="Куріння")

    plt.show()

if __name__ == "__main__":
    main()
