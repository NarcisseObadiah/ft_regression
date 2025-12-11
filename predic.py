import json
import sys

# ---------------------------------------------------------
# Load saved thetas
# ---------------------------------------------------------
def load_thetas(filename="thetas.json"):
    """
    Load theta0 and theta1 from JSON file.
    These thetas are expected to be denormalized (real scale)
    and therefore accept raw mileage (km) as input.
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'thetas.json' not found. Run training first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: 'thetas.json' contains invalid JSON.")
        sys.exit(1)

    # Validate presence of required keys
    try:
        theta0 = float(data["theta0"])
        theta1 = float(data["theta1"])
    except KeyError:
        print("Error: 'thetas.json' missing keys 'theta0' and/or 'theta1'.")
        sys.exit(1)
    except (TypeError, ValueError):
        print("Error: Invalid theta values in 'thetas.json'.")
        sys.exit(1)

    return theta0, theta1


# ---------------------------------------------------------
# Prediction logic (uses raw mileage — NOT normalized)
# ---------------------------------------------------------
def estimate_price(mileage, theta0, theta1):
    """
    Estimate the price given raw mileage (in km) using:
        price = theta0 + theta1 * mileage
    theta0 and theta1 must be on the same (raw) scale.
    """
    return theta0 + theta1 * mileage


# ---------------------------------------------------------
# CLI mode
# ---------------------------------------------------------
if __name__ == "__main__":
    theta0, theta1 = load_thetas()

    # Ask user for mileage
    try:
        mileage_input = input("Enter car mileage (km): ")
        mileage = float(mileage_input)
    except ValueError:
        print("Invalid mileage value.")
        sys.exit(1)

    price = estimate_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f} €")
