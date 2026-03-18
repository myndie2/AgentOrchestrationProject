# Agent qui orchestre tout

from ingredient_agent import ingredient_agent
from recipe_agent import recipe_agent
from nutrition_agent import nutrition_agent
from diet_agent import diet_agent
from improvement_agent import improvement_agent


def manager_agent(ingredients):

    analysis = ingredient_agent(ingredients)

    recipe = recipe_agent(ingredients, analysis)

    nutrition = nutrition_agent(recipe)

    diet = diet_agent(recipe)

    improvement = improvement_agent(recipe)

    return analysis, recipe, nutrition, diet, improvement