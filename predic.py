import json

def load_thetas(filename="thetas.json"):
    try:
        with open(filename, "r") as file:
            data = json.load(file)
        if "theta0" not in data or "theta1" not in data:
            raise ValueError("Invalid thetas.json format")

        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])

    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Warning: {e}. Please launch training before any prediction!!!")
        theta0, theta1 = 0.0, 0.0

    return theta0, theta1


# prediction logic-- > f(x) = b + wxi --> price =  theta0 + theta1 * mileage
def estimate_price(mileage, theta0, theta1):
    return theta0 + theta1 * mileage


if __name__ == "__main__":
    theta0, theta1 = load_thetas()
    try:
        mileage = float(input("Enter car mileage (km): "))
    except ValueError:
        print("Invalid mileage input, please enter a valid input... :)")
        exit(1)
    price = estimate_price(mileage, theta0, theta1)
    print(f"This car price is estimated around : {price:.2f} €")
