import pandas as pd

def main():
    file_path = input("Enter CSV file path: ").strip()

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print("\nColumns in dataset:\n")
    for i, col in enumerate(df.columns):
        print(f"{i}: {col}")

    print("\nSelect columns to remove:")
    print("Options:")
    print("1. Enter column numbers (comma-separated)")
    print("2. Enter column names (comma-separated)")

    choice = input("\nYour choice (1 or 2): ").strip()

    cols_to_drop = []

    if choice == "1":
        indices = input("Enter column indices (e.g., 0,3,5): ").strip().split(",")
        try:
            cols_to_drop = [df.columns[int(i.strip())] for i in indices]
        except Exception:
            print("Invalid indices entered.")
            return

    elif choice == "2":
        names = input("Enter column names (comma-separated): ").strip().split(",")
        cols_to_drop = [name.strip() for name in names]

    else:
        print("Invalid choice.")
        return

    print("\nColumns to be removed:")
    for col in cols_to_drop:
        print(f"- {col}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    df_cleaned = df.drop(columns=cols_to_drop, errors="ignore")

    output_path = input("\nEnter output file name (default: cleaned.csv): ").strip()
    if not output_path:
        output_path = "cleaned.csv"

    df_cleaned.to_csv(output_path, index=False)

    print(f"\n✅ Cleaned dataset saved as: {output_path}")


if __name__ == "__main__":
    main()