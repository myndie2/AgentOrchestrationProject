from manager_agent import manager_agent


def main():

    ingredients = input("Enter ingredients: ")

    analysis, recipe, nutrition, diet, improvement = manager_agent(ingredients)

    print("\n--- Ingredient analysis ---")
    print(analysis)

    print("\n--- Recipe generation ---")
    print(recipe)

    print("\n--- Nutrition estimation ---")
    print(nutrition)

    print("\n--- Diet analysis ---")
    print(diet)

    print("\n--- Recipe improvement ---")
    print(improvement)


if __name__ == "__main__":
    main()