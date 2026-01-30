
# ---- Settings ----
data_path <- "data/EggSizeData.csv"
seed <- 123
train_prop <- 0.8

# ---- 1) Load + clean ----
dat <- load_and_clean_egg_data(data_path)

cat("\nRows after cleaning:", nrow(dat), "\n")
cat("Class counts:\n")
print(table(dat$Species))

# ---- 2) Split + train ----
spl <- split_train_test(dat, prop = train_prop, seed = seed)
lda_model <- fit_lda_model(spl$train)

cat("\nLDA model:\n")
print(lda_model)

# ---- 3) Evaluate ----
ev <- evaluate_lda(lda_model, spl$test)

cat("\nConfusion matrix:\n")
print(ev$confusion_matrix)

cat("\nAccuracy:", round(ev$accuracy, 4), "\n")

# ---- 4) Save plots ----
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)

plots <- plot_actual_vs_predicted(spl$test, ev$predictions$class)

ggplot2::ggsave(
  filename = "outputs/figures/actual_species.png",
  plot = plots$actual_plot,
  width = 7, height = 5, dpi = 200
)

ggplot2::ggsave(
  filename = "outputs/figures/predicted_species.png",
  plot = plots$predicted_plot,
  width = 7, height = 5, dpi = 200
)

ggplot2::ggsave(
  filename = "outputs/figures/lda_boundary.png",
  plot = plot_lda_decision_boundary(lda_model, spl$test),
  width = 7, height = 5, dpi = 200
)

cat("\nSaved figures to outputs/figures/\n")

# ---- 5) Optional map ----
if (all(c("lat", "long", "loc") %in% names(dat))) {
  dir.create("outputs/maps", recursive = TRUE, showWarnings = FALSE)
  make_location_map(dat, out_html = "outputs/maps/egg_map.html")
  cat("Saved map to outputs/maps/egg_map.html\n")
} else {
  cat("Map skipped: lat/long/loc columns not found.\n")
}

# ---- Example prediction ----
example_data <- data.frame(LENGTH = 75, WIDTH = 50)
pred_ex <- predict(lda_model, newdata = example_data)

cat("\nExample input:\n")
print(example_data)

cat("\nPredicted class:\n")
print(pred_ex$class)

cat("\nPosterior probabilities:\n")
print(pred_ex$posterior)

# R/01_load_clean.R
suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

# Convert raw species labels into two classes:
# - Cackling Goose (includes your B.h... codes)
# - Canada Goose (everything else)
recode_species <- function(x) {
  x <- as.character(x)
  
  cackling_codes <- c(
    "B.h.hutchinsii",
    "B.hutchinsii",
    "B.hutchinsii.minima",
    "Cackling Goose"
  )
  
  ifelse(x %in% cackling_codes, "Cackling Goose", "Canada Goose")
}

load_and_clean_egg_data <- function(path,
                                    length_range = c(50, 100),
                                    width_range  = c(30, 70)) {
  
  dat <- readr::read_csv(path, show_col_types = FALSE)
  
  required <- c("LENGTH", "WIDTH", "Species")
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) {
    stop("Missing required columns: ", paste(missing, collapse = ", "))
  }
  
  # Convert columns safely (if present)
  dat <- dat %>%
    mutate(
      LENGTH  = as.numeric(LENGTH),
      WIDTH   = as.numeric(WIDTH),
      MASS    = if ("MASS" %in% names(.)) as.numeric(MASS) else NA_real_,
      YEAR    = if ("YEAR" %in% names(.)) as.factor(YEAR) else NA,
      loc     = if ("loc" %in% names(.)) as.factor(loc) else NA,
      EGGNUM  = if ("EGGNUM" %in% names(.)) as.factor(EGGNUM) else NA,
      lat     = if ("lat" %in% names(.)) as.numeric(lat) else NA_real_,
      long    = if ("long" %in% names(.)) as.numeric(long) else NA_real_,
      Species = factor(recode_species(Species))
    ) %>%
    filter(!is.na(LENGTH), !is.na(WIDTH)) %>%
    filter(
      dplyr::between(LENGTH, length_range[1], length_range[2]),
      dplyr::between(WIDTH,  width_range[1],  width_range[2])
    )
  
  # Basic sanity check
  if (nlevels(dat$Species) < 2) {
    warning("After cleaning/recode, Species has <2 classes. Check Species labels.")
  }
  
  dat
}

# R/02_train_evaluate_lda.R
suppressPackageStartupMessages({
  library(MASS)
})

split_train_test <- function(dat, prop = 0.8, seed = 123) {
  set.seed(seed)
  idx <- sample(seq_len(nrow(dat)), size = floor(prop * nrow(dat)))
  list(
    train = dat[idx, , drop = FALSE],
    test  = dat[-idx, , drop = FALSE]
  )
}

fit_lda_model <- function(train_dat) {
  MASS::lda(Species ~ LENGTH + WIDTH, data = train_dat)
}

evaluate_lda <- function(model, test_dat) {
  pred <- predict(model, newdata = test_dat)
  cm <- table(Predicted = pred$class, Actual = test_dat$Species)
  acc <- sum(diag(cm)) / sum(cm)
  
  list(
    confusion_matrix = cm,
    accuracy = acc,
    predictions = pred
  )
}

# R/03_visualize_lda.R
suppressPackageStartupMessages({
  library(ggplot2)
  library(rlang)
  library(dplyr)
})

plot_actual_vs_predicted <- function(test_dat, pred_class) {
  p1 <- ggplot(test_dat, aes(x = LENGTH, y = WIDTH, color = Species)) +
    geom_point(alpha = 0.8) +
    ggtitle("Actual Species") +
    theme_minimal()
  
  tmp <- test_dat %>% mutate(Predicted = pred_class)
  
  p2 <- ggplot(tmp, aes(x = LENGTH, y = WIDTH, color = Predicted)) +
    geom_point(alpha = 0.8) +
    ggtitle("Predicted Species (LDA)") +
    theme_minimal()
  
  list(actual_plot = p1, predicted_plot = p2)
}

plot_lda_decision_boundary <- function(model, dat, grid_n = 200) {
  xseq <- seq(min(dat$LENGTH, na.rm = TRUE), max(dat$LENGTH, na.rm = TRUE), length.out = grid_n)
  yseq <- seq(min(dat$WIDTH,  na.rm = TRUE), max(dat$WIDTH,  na.rm = TRUE), length.out = grid_n)
  
  grid <- expand.grid(LENGTH = xseq, WIDTH = yseq)
  grid$Pred <- predict(model, newdata = grid)$class
  
  ggplot(dat, aes(LENGTH, WIDTH, color = Species)) +
    geom_point(alpha = 0.7) +
    geom_raster(
      data = grid,
      aes(LENGTH, WIDTH, fill = Pred),
      alpha = 0.2,
      inherit.aes = FALSE
    ) +
    ggtitle("LDA Decision Regions (LENGTH vs WIDTH)") +
    theme_minimal()
}

# R/04_map_leaflet.R
suppressPackageStartupMessages({
  library(dplyr)
  library(leaflet)
  library(htmlwidgets)
})

make_location_map <- function(dat, out_html = "outputs/maps/egg_map.html") {
  needed <- c("lat", "long", "loc", "Species")
  miss <- setdiff(needed, names(dat))
  if (length(miss) > 0) {
    stop("Missing columns for mapping: ", paste(miss, collapse = ", "))
  }
  
  mdat <- dat %>%
    filter(!is.na(lat), !is.na(long), !is.na(loc)) %>%
    group_by(lat, long, loc, Species) %>%
    summarise(count = n(), .groups = "drop")
  
  pal <- colorFactor(palette = "Dark2", domain = mdat$loc)
  
  m <- leaflet(mdat) %>%
    addTiles() %>%
    addCircleMarkers(
      lng = ~long, lat = ~lat,
      color = ~pal(loc),
      radius = ~pmax(3, sqrt(count)),
      popup = ~paste0(
        "Location: ", loc, "<br>",
        "Species: ", Species, "<br>",
        "Count: ", count, "<br>",
        "Lat: ", lat, "<br>",
        "Long: ", long
      )
    ) %>%
    addLegend("bottomright", pal = pal, values = ~loc, title = "Location", opacity = 1)
  
  dir.create(dirname(out_html), recursive = TRUE, showWarnings = FALSE)
  saveWidget(m, out_html, selfcontained = TRUE)
  invisible(m)
}




