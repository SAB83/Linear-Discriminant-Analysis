# Egg Size LDA Classifier (Canada vs Cackling Goose)

This project classifies goose species using egg measurements (**LENGTH** and **WIDTH**) with **Linear Discriminant Analysis (LDA)** in R.

It:
- Cleans the dataset (removes NAs, filters unrealistic egg sizes)
- Recodes species into **two classes**: Canada Goose vs Cackling Goose
- Trains an LDA model and evaluates accuracy on a test split
- Saves plots and (optionally) an interactive Leaflet map
